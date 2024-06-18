import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict
import pytorch_lightning as pl
import torch.nn.functional as F
from protonmof.utils import calculate_con, arrhenius_reg, NoamLR
from torch.optim import AdamW
from typing import List
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error



class MOFProtonModel(pl.LightningModule):
    def __init__(self, config: EasyDict, 
                scaler = None):
        """
        param config: EasyDict object which includes information about model parameters.
        """
        super().__init__()
        self.config = config
        self.arr = config.model.arr # 'arrhenius',  None
        assert self.arr in ['arrhenius',  'None']
        self.arr_reg_rate = config.model.arr_reg_rate
        self.ea_reg_rate = config.model.ea_reg_rate
        self.desc_dim = config.model.desc_dim
        self.hidden_dim = config.model.hidden_dim
        self.num_layer = config.model.num_layer
        self.exp_name = config.train.exp_name
        self.rmse = config.model.rmse
        self.scaler = scaler
        
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        if config.model.act == 'silu':
            self.act = nn.SiLU()
        elif config.model.act == 'relu':
            self.act = nn.ReLU()
        
        if self.arr == 'arrhenius':
            self.output_dim = 2

        else:
            self.output_dim = 1



        self.mof_dense = nn.Linear(self.desc_dim, self.hidden_dim )
        self.guest_dense = nn.Linear(self.desc_dim, self.hidden_dim )
        self.t_dense = nn.Linear(1, self.hidden_dim )
        self.rh_dense = nn.Linear(1, self.hidden_dim)
        
        mlp= [
            
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            self.act,
            nn.Dropout(0.5)
        ]
        for _ in range(self.num_layer-1):
            mlp.extend([
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                self.act,
                nn.Dropout(0.5)
            ])

        mlp.extend([
            nn.Linear(self.hidden_dim, self.output_dim),

        ])

        self.mlp = nn.Sequential(*mlp)

    def forward(self,
                batch: List[np.ndarray]):
        """
        param batch: MOF, guest, log(proton con), T+RH
        
        return output
        self.arr -> None | shape: [B, 1]
        self.arr -> arrhenius | shape: [B, 2]
        """
        mof_embeds = self.mof_dense(batch['mof_descriptors'])
        guest_embeds = self.guest_dense(batch['guest_descriptors'])
        t_embeds = self.t_dense(batch['T'])
        rh_embeds = self.rh_dense(batch['RH'])
        if (self.arr == 'arrhenius') :
            all_embeds = mof_embeds + guest_embeds + rh_embeds
        else:
            all_embeds = mof_embeds + guest_embeds + t_embeds + rh_embeds
        
        output = self.mlp(all_embeds)        


        return output


            
    def training_step(self, batch, batch_idx):
        output  = self(batch)
        DEVICE = output.device
        con_target = batch['proton_conductivity'].reshape(-1).to(DEVICE)

        # calculate conductivity from Ea, logA, T, T0, etc using equations.
        con_pred = calculate_con(output, batch, self.arr)
        reg_term = arrhenius_reg(output, self.arr, reg_rate = self.arr_reg_rate)
        total_loss = self.regression_loss(con_pred, con_target, rmse= self.rmse) + reg_term 

        self.log('train_loss', total_loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist = True)
        #print(total_loss, total_loss.item())
        return total_loss

    
    def validation_step(self, batch,batch_idx ):
        output  = self(batch)
        DEVICE = output.device
        con_target = batch['proton_conductivity'].reshape(-1).to(DEVICE)

        # calculate conductivity from Ea, logA, T, T0, etc using equations.
        con_pred = calculate_con(output, batch, self.arr)
        reg_term = arrhenius_reg(output, self.arr, reg_rate = self.arr_reg_rate)
        total_loss = self.regression_loss(con_pred, con_target, rmse= self.rmse) + reg_term 
        #print(total_loss, batch_idx)
            
        self.log('val_loss', total_loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist = True)
        output_dict = {'val_loss': total_loss.item(), 'batch_size': len(con_target), 'pred': con_pred.cpu().tolist(), 'target': con_target.cpu().tolist()}
        self.validation_step_outputs.append(output_dict)
        return  output_dict

    
    def on_validation_epoch_end(self):
        
        outputs = self.validation_step_outputs
        loss_sum = np.array([x['val_loss'] * x['batch_size'] for x in outputs])
        avg_loss = np.sum(loss_sum)/np.sum([x['batch_size'] for x in outputs])
        all_target = np.concatenate([x['target'] for x in outputs])
        all_pred = np.concatenate([x['pred'] for x in outputs])

        if self.scaler is not None:
            all_target = self.scaler.decode(all_target)
            all_pred = self.scaler.decode(all_pred )        
        
        self.log('avg_r2_score', r2_score(all_target, all_pred))
        self.log('avg_mae_loss', mean_absolute_error(all_target, all_pred))        
        self.log('avg_val_loss', avg_loss)
        self.validation_step_outputs.clear()


    def test_step(self, batch, batch_idx):
        output  = self(batch)
        DEVICE = output.device
        con_target = batch['proton_conductivity'].reshape(-1).to(DEVICE)

        # calculate conductivity from Ea, logA, T, T0, etc using equations.
        con_pred = calculate_con(output, batch, self.arr)
        reg_term = arrhenius_reg(output, self.arr, reg_rate = self.arr_reg_rate)

        total_loss = self.regression_loss(con_pred, con_target, rmse= self.rmse) + reg_term 
        output = [con_pred.cpu().tolist(), con_target.cpu().tolist(), total_loss.cpu().tolist()]
        self.test_step_outputs.append(output)
        return output
    
    def on_test_epoch_end(self, ):

        logits, labels, losses,batch_size  =  [], [], [],[]
        outputs = self.test_step_outputs
        for pred, target, loss in outputs:

            logits += list(np.array(pred).reshape(-1))
            labels += list(np.array(target).reshape(-1))
            losses.append(loss*len(target))
            batch_size.append(len(target))
          

        logits = np.array(logits)
        labels = np.array(labels)
        if self.scaler is not None:
            logits = self.scaler.decode(logits )
            labels = self.scaler.decode(labels )

        mse_loss = mean_squared_error(labels, logits)
        rmse_loss = mean_squared_error(labels, logits, squared=False)
        r2= r2_score( labels, logits)

        np.savez(f'{self.exp_name}_logits_labels.npz', logits=logits, labels=labels)
        print('mse', mse_loss)        
        print('rmse',rmse_loss )
        print('mae', mean_absolute_error(labels,logits))
        print('r2', r2)
        print('loss', np.sum(losses)/np.sum(batch_size))
        self.test_step_outputs.clear()
        return {'logits':logits, 'labels': labels}


    
    def regression_loss(self, preds, target, rmse= False):
        loss = F.mse_loss(preds, target, reduction='mean')
        if rmse:
            eps = 1e-7
            loss = torch.sqrt(loss+eps)
        
        return loss 


    def infer_last_layer(self,batch):
        mof_embeds = self.mof_dense(batch['mof_descriptors'])
        guest_embeds = self.guest_dense(batch['guest_descriptors'])
        t_embeds = self.t_dense(batch['T'])
        rh_embeds = self.rh_dense(batch['RH'])
        if (self.arr == 'arrhenius') or (self.arr == 'vtf'):
            all_embeds = mof_embeds + guest_embeds + rh_embeds
        else:
            all_embeds = mof_embeds + guest_embeds + t_embeds + rh_embeds
        
        output = self.mlp[:-1](all_embeds)        


        return output        

    
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() ],
                "weight_decay": float(self.config.train.weight_decay),
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=float(self.config.train.init_lr))
        

        scheduler = NoamLR(
            optimizer=optimizer,
            warmup_epochs=[int(self.config.train.warmup_epochs)],
            total_epochs=[int(self.config.train.epochs)],
            steps_per_epoch=int(self.config.train.train_data_size) // int(self.config.train.batch_size),
            init_lr=[float(self.config.train.init_lr)],
            max_lr=[float(self.config.train.max_lr)],
            final_lr=[float(self.config.train.final_lr)]
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]