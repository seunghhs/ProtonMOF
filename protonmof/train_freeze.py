import os
import shutil
import argparse
import yaml
from easydict import EasyDict
from tqdm.auto import tqdm
from glob import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

from protonmof.data.dataset import MOFProtonDataset
from protonmof.utils import split_by_cif, Normalizer
from protonmof.model.model import MOFProtonModel
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

torch.multiprocessing.set_sharing_strategy("file_system")

def main(args, fold):
    pl.seed_everything(config.train.seed)
    os.makedirs(config.train.log_dir, exist_ok=True)

    ckpt_dir = f'./ckpt_{config.train.exp_name}_{fold}/'
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(

        dirpath = ckpt_dir,
        save_top_k=1,
        verbose=True,
        save_last=True,
        monitor="avg_mae_loss",
        mode='min'
    )    

    logger = pl.loggers.TensorBoardLogger(
        config.train.log_dir,
        name=f'{config.train.exp_name}_seed{config.train.seed}/',
    )        

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    early_callback = EarlyStopping(monitor="avg_mae_loss", mode="min",patience=10,)

    callbacks = [checkpoint_callback, lr_callback, early_callback]




    num_nodes = config.train.num_nodes



    log_every_n_steps=1

    train_data= MOFProtonDataset(proton_data = train_data_tmp,
                                     config=config,
                                 scaler=prop_scaler,
                                 t_scaler = t_scaler,
                                 rh_scaler = rh_scaler,
  
                                      )
    config.train.train_data_size =len(train_data)


    #model

    model = MOFProtonModel(config,scaler = prop_scaler,)
    trainer = Trainer(

                      accelerator = args.accelerator,
                      devices = args.devices,
                      num_nodes = config.train.num_nodes,
                      max_epochs=config.train.epochs, 
                      #deterministic=True,
                      logger=logger,
                      benchmark=True,
                      strategy='ddp',
                      #resume_from_checkpoint= config.train.resume_from,
                       log_every_n_steps=log_every_n_steps,
                      callbacks=callbacks
                     )


    valid_data = MOFProtonDataset(proton_data = valid_data_tmp,
                                     config=config,
                                  scaler = prop_scaler,
                                 t_scaler = t_scaler,
                                 rh_scaler = rh_scaler,
                              
                                      )


    train_loader =DataLoader(train_data, config.train.batch_size , num_workers = config.train.num_workers, 
                             shuffle=True)    




    valid_loader =DataLoader(valid_data, config.train.batch_size , num_workers = config.train.num_workers,
                             shuffle=False)        



    if args.test:
        
        #test_data_tmp.to_excel('tmp.xlsx', index=None)     
        test_data = MOFProtonDataset(proton_data = test_data_tmp,
                                 config=config,
                                     scaler = prop_scaler,
                                 t_scaler = t_scaler,
                                 rh_scaler = rh_scaler,
                                     
                                  )

        test_loader =DataLoader(test_data, config.train.batch_size , num_workers = config.train.num_workers, 
                                 shuffle=False)   
  

        if config.test.ckpt_path != 'None':
            tmp_dir = config.test.ckpt_path
        else:
            tmp_dir = ckpt_dir    
        files=glob(f'{tmp_dir}/*.ckpt')
        for f in files:
            name=f.split('/')[-1]
            if name.startswith('epoch'):
                target_path = f        

        print(target_path)
        model = MOFProtonModel.load_from_checkpoint(target_path,  config=config, scaler = prop_scaler, strict=False) 
        trainer.test(model, train_loader)
        logits_labels = np.load(f'{config.train.exp_name}_logits_labels.npz')
        logits = logits_labels['logits']
        labels = logits_labels['labels']        
        np.savez(f'{config.train.exp_name}_logits_labels_train.npz', logits=logits, labels=labels)
        

        trainer.test(model, valid_loader)
        logits_labels = np.load(f'{config.train.exp_name}_logits_labels.npz')
        logits = logits_labels['logits']
        labels = logits_labels['labels']        
        np.savez(f'{config.train.exp_name}_logits_labels_valid.npz', logits=logits, labels=labels)


        trainer.test(model, test_loader)
        logits_labels = np.load(f'{config.train.exp_name}_logits_labels.npz')
        logits = logits_labels['logits']
        labels = logits_labels['labels']        
        np.savez(f'{config.train.exp_name}_logits_labels_test.npz', logits=logits, labels=labels)
        

        
        
    else:



        trainer.fit(model, train_loader, valid_loader)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default = 1)
    parser.add_argument('--resume_iter', type=int, default=None)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--kfold', type=int, default=1)

    args = parser.parse_args()

    
    resume = os.path.isdir(args.config)
    if resume:
        config_path = glob(os.path.join(args.config, '*.yml'))[0]
        resume_from = args.config
    else:
        config_path = args.config

    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    #scaler for target(proton conductivity) value

    if config.dataset.prop_scaler:
        prop_scaler = Normalizer(mean = config.dataset.mean, std = config.dataset.std)
    else:
        prop_scaler = None

    t_scaler = None
    rh_scaler = None    
    if config.dataset.t_scaler:
        t_scaler = Normalizer(mean = config.dataset.t_mean, std = config.dataset.t_std)
    if config.dataset.rh_scaler:
        rh_scaler = Normalizer(mean = config.dataset.rh_mean, std = config.dataset.rh_std)



    
    train_data_tmp = pd.read_csv(config.dataset.train_data_path)
    valid_data_tmp = pd.read_csv(config.dataset.valid_data_path)
    test_data_tmp = pd.read_csv(config.dataset.test_data_path)


    
    all_data = pd.concat([train_data_tmp, valid_data_tmp], axis=0)
    
    if args.kfold >1 :        
        splits = KFold(n_splits=args.kfold, shuffle=True,random_state=0)
        test_r2_score, test_rmse_score, = [], []
        for fold, (train_idx, valid_idx) in enumerate(list(splits.split(np.arange(len(all_data))))):
            train_data_tmp = all_data.iloc[train_idx]
            valid_data_tmp = all_data.iloc[valid_idx]
            main(args, fold)

    else:
        main(args, 0)
        

