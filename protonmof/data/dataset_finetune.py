import os
import numpy as np
import pandas as pd
import torch
import json
#from torch.utils.data import DataLoader, Dataset
from easydict import EasyDict
from protonmof.utils import convert_none_type
from typing import List, Union, Optional, Any
from moftransformer.datamodules import dataset
from simpletransformers.classification import ClassificationModel, ClassificationArgs


class MOFProtonDataset(dataset.Dataset):
    """A :class:`MOFProtonDataset` contains MOFdescriptors + guest_descriptors."""

    def __init__(self, 
                 proton_data: pd.DataFrame,

                 config: EasyDict,
                 scaler: Optional[Any] = None,
                 t_scaler: Optional[Any] = None,
                 rh_scaler: Optional[Any] = None,
                draw_false_grid=False,
                 nbr_fea_len = 64,
                 
                ):
        r"""
        :param proton_data: it is pd.DataFrame whith contains DOI, proton conductivity, name(csd identifier), temperature, RH, Ea, Guest.
        param mof_desc_path: MOF Descriptors dataframe path
        param guest_desc_path: Guest Descriptors dataframe path
        """

        self.data_dir = config.dataset.mof_data_dir
        self.guest_data_dir = config.dataset.guest_data_dir
        self.split = ''
        self.draw_false_grid = draw_false_grid
        self.nbr_fea_len = nbr_fea_len
        self.tasks = {}
        
        cif_list = sorted(list(set([f.split('.')[0] for f in os.listdir(self.data_dir) if f.split('.')[0]])))
        self.data = self.drop_no_MOF_desc(proton_data,cif_list )


        self.con, self.T,self.RH, self.smiles, self.smiles2, self.cif_ids =self.get_info()


        self.arr = convert_none_type(config.model.arr)
        self.scaler = scaler
        self.t_scaler = t_scaler
        self.rh_scaler = rh_scaler

        if self.scaler is not None:
            self.con = self.scaler.encode(self.con)

        if self.t_scaler is not None:
            self.T = self.t_scaler.encode(self.T)
            
        if self.rh_scaler is not None:
            self.RH = self.rh_scaler.encode(self.RH)
    
        #self.T = self.T.reshape(-1,1)
        #self.RH = self.RH.reshape(-1,1)

        self.tokenizer = ClassificationModel('roberta', 'seyonec/PubChem10M_SMILES_BPE_396_250', 
                                    num_labels=1,
                                    args={'evaluate_each_epoch': True, 
                                          'evaluate_during_training_verbose': True,
                                          'no_save': False, 'num_train_epochs': 10, 
                                          'regression' : True,
                                          'auto_weights': True}).tokenizer 
        
        self.input_ids, self.attention_mask = self.get_tokens(self.smiles)
        self.input_ids2, self.attention_mask2 = self.get_tokens(self.smiles2)
    
    def __len__(self):
        return len(self.con)

    def __getitem__(self, index):
        ret = dict()
        cif_id = self.cif_ids[index]
        ret.update(
            {
                "cif_id": self.cif_ids[index],
                "proton_conductivity": self.con[index],
                "T": self.T[index],
                "RH": self.RH[index],
                "input_ids": self.input_ids[index],
                "attention_mask": self.attention_mask[index],
                "input_ids2": self.input_ids2[index],
                "attention_mask2": self.attention_mask2[index],                
            }
        )
        ret.update(self.get_grid_data(cif_id, draw_false_grid=self.draw_false_grid))
        ret.update(self.get_graph(cif_id))

        ret.update(self.get_tasks(index))  



        return ret

    
        
    def get_info(self):
        all_con, all_T, all_RH, all_smiles, all_smiles2, all_cif = [], [], [], [], [], []
        with open(self.guest_data_dir, 'r') as f:
            smiles_dict = json.load(f)

        for i, row in self.data.iterrows():
            T = row['Temperature']
            RH = row['RH']
            con = row['proton conductivity']
            cif = row['Name']
            
            if pd.isna(row['Guest']):
                smiles = ''
                smiles2 = ''
            else:
                guest_list = row['Guest'].split(',')
                if len(guest_list) == 1:
                    smiles = smiles_dict[row['Guest']]
                    smiles2 = ''
                else:
                    smiles, smiles2 = [smiles_dict[guest] for guest in guest_list]
                    
            all_con.append(np.log10(con))
            all_T.append(T)
            all_RH.append(RH)
            all_smiles.append(smiles)
            all_smiles2.append(smiles2)
            all_cif.append(cif)

        all_con = torch.tensor(all_con, dtype=torch.float32)
        all_T = torch.tensor(all_T, dtype=torch.float32)
        all_RH = torch.tensor(all_RH, dtype=torch.float32)

        

        return all_con, all_T, all_RH, all_smiles, all_smiles2, all_cif

                            
            
    
    def drop_no_MOF_desc(self,data, cif_ids):
        """
        drop rows which doesn't have MOF descriptors in the data (DataFrame)
        """
        drop_index =[]
        drop_set=set()
        for row in data.iterrows():
            if row[1]['Name'] not in cif_ids:
                drop_index.append(row[0])
                drop_set.add(row[1]['Name'])
        
        new_data=data.drop(drop_index, axis=0)
        new_data = new_data.reset_index(drop=True)
        return new_data



    def get_tokens(self, smiles):
        tokens = self.tokenizer(smiles,add_special_tokens=True, truncation=True, 
                                         max_length=256, padding="max_length", 
                                      return_tensors='pt',
                                      return_offsets_mapping=False)
        for k, v in tokens.items():
            tokens[k] = torch.tensor(v, dtype=torch.long,)   

        return tokens['input_ids'], tokens['attention_mask']


    @staticmethod
    def collate(batch, img_size):
        """
        collate batch
        Args:
            batch (dict): [cif_id, atom_num, nbr_idx, nbr_fea, uni_idx, uni_count,
                            grid_data, cell, (false_grid_data, false_cell), target]
            img_size (int): maximum length of img size

        Returns:
            dict_batch (dict): [cif_id, atom_num, nbr_idx, nbr_fea, crystal_atom_idx,
                                uni_idx, uni_count, grid, false_grid_data, target]
        """
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])

        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        # graph
        batch_atom_num = dict_batch["atom_num"]
        batch_nbr_idx = dict_batch["nbr_idx"]
        batch_nbr_fea = dict_batch["nbr_fea"]

        crystal_atom_idx = []
        base_idx = 0
        for i, nbr_idx in enumerate(batch_nbr_idx):
            n_i = nbr_idx.shape[0]
            crystal_atom_idx.append(torch.arange(n_i) + base_idx)
            nbr_idx += base_idx
            base_idx += n_i

        dict_batch["atom_num"] = torch.cat(batch_atom_num, dim=0)
        dict_batch["nbr_idx"] = torch.cat(batch_nbr_idx, dim=0)
        dict_batch["nbr_fea"] = torch.cat(batch_nbr_fea, dim=0)
        dict_batch["crystal_atom_idx"] = crystal_atom_idx
        dict_batch['T'] = torch.stack(dict_batch['T'] , dim=0).reshape(-1,1)
        dict_batch['RH'] = torch.stack(dict_batch['RH'] , dim=0).reshape(-1,1)
        dict_batch['proton_conductivity'] = torch.stack(dict_batch['proton_conductivity'] , dim=0)
        dict_batch['input_ids'] = torch.stack(dict_batch['input_ids'])
        dict_batch['attention_mask'] = torch.stack(dict_batch['attention_mask'])
        dict_batch['input_ids2'] = torch.stack(dict_batch['input_ids2'])
        dict_batch['attention_mask2'] = torch.stack(dict_batch['attention_mask2'])        

        # grid
        batch_grid_data = dict_batch["grid_data"]
        batch_cell = dict_batch["cell"]
        new_grids = []

        for bi in range(batch_size):
            orig = batch_grid_data[bi].view(batch_cell[bi][::-1]).transpose(0, 2)
            if batch_cell[bi] == [30, 30, 30]:  # version >= 1.1.2
                orig = orig[None, None, :, :, :]
            else:
                orig = interpolate(
                    orig[None, None, :, :, :],
                    size=[img_size, img_size, img_size],
                    mode="trilinear",
                    align_corners=True,
                )
            new_grids.append(orig)
        new_grids = torch.concat(new_grids, axis=0)
        dict_batch["grid"] = new_grids

        if "false_grid_data" in dict_batch.keys():
            batch_false_grid_data = dict_batch["false_grid_data"]
            batch_false_cell = dict_batch["false_cell"]
            new_false_grids = []
            for bi in range(batch_size):
                orig = batch_false_grid_data[bi].view(batch_false_cell[bi])
                if batch_cell[bi] == [30, 30, 30]:  # version >= 1.1.2
                    orig = orig[None, None, :, :, :]
                else:
                    orig = interpolate(
                        orig[None, None, :, :, :],
                        size=[img_size, img_size, img_size],
                        mode="trilinear",
                        align_corners=True,
                    )
                new_false_grids.append(orig)
            new_false_grids = torch.concat(new_false_grids, axis=0)
            dict_batch["false_grid"] = new_false_grids

        dict_batch.pop("grid_data", None)
        dict_batch.pop("false_grid_data", None)
        dict_batch.pop("cell", None)
        dict_batch.pop("false_cell", None)

        return dict_batch
    