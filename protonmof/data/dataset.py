import numpy as np
import pandas as pd
import torch
import json
from torch.utils.data import DataLoader, Dataset
from easydict import EasyDict
from protonmof.utils import convert_none_type
from typing import List, Union, Optional, Any

class MOFProtonDataset(Dataset):
    """A :class:`MOFProtonDataset` contains MOFdescriptors + guest_descriptors."""

    def __init__(self, 
                 proton_data: pd.DataFrame,
                 config: EasyDict,
                 scaler: Optional[Any] = None,
                 t_scaler: Optional[Any] = None,
                 rh_scaler: Optional[Any] = None,

                 
                ):
        r"""
        :param proton_data: it is pd.DataFrame whith contains DOI, proton conductivity, name(csd identifier), temperature, RH, Ea, Guest.
        param mof_desc_path: MOF Descriptors dataframe path (MOFTransformer)
        param guest_desc_path: Guest Descriptors dataframe path (ChemBERT)
        """

        with open(config.dataset.mof_desc_path, 'r') as f:
            mof_desc_dict = json.load(f)
        
        
        with open(config.dataset.guest_desc_path, 'r') as f:
            guest_desc_dict = (json.load(f))
                

        self.data = self.drop_no_MOF_desc(proton_data,mof_desc_dict )


        self.con, self.T,self.RH, self.mof_desc, self.guest_desc, self.cif_ids = self.extract_descriptors_moftrans_chembert(self.data,
                             mof_desc_dict = mof_desc_dict, guest_desc_dict=guest_desc_dict,
                              )  


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
    
        self.T = self.T.reshape(-1,1)
        self.RH = self.RH.reshape(-1,1)
    
    def __len__(self):
        return len(self.con)

    def __getitem__(self, idx):
        """

        """        



        return {'proton_conductivity': self.con[idx], 'T': self.T[idx], 'RH': self.RH[idx],'cif_id': self.cif_ids[idx],
                'mof_descriptors': self.mof_desc[idx], 'guest_descriptors': self.guest_desc[idx] }

    
        

    def drop_no_MOF_desc(self,data, mof_desc_dict):
        """
        drop rows which doesn't have MOF descriptors in the data (DataFrame)
        """
        drop_index =[]
        drop_set=set()
        for row in data.iterrows():
            if row[1]['Name'] not in mof_desc_dict.keys():
                drop_index.append(row[0])
                drop_set.add(row[1]['Name'])
        
        new_data=data.drop(drop_index, axis=0)
        new_data = new_data.reset_index(drop=True)
        return new_data



    def extract_descriptors_moftrans_chembert(self, data, 
                            mof_desc_dict,  guest_desc_dict,
                            
                             ):
        all_con, all_T, all_RH, all_mof_desc, all_guest_desc, all_cif = [], [], [], [], [], []
        g_num = len(list(guest_desc_dict.values())[0])
        for i, row in data.iterrows():
            T = row['Temperature']
            RH = row['RH']
            con = row['proton conductivity']
            cif = row['Name']
            mof_desc = mof_desc_dict[cif]
            
            if pd.isna(row['Guest']):
                guest_desc = [0 for _ in range(g_num)]
            else:
                guest_list = row['Guest'].split(',')
                if len(guest_list) == 1:
                    guest_desc = guest_desc_dict[row['Guest']]
                else:
                    guest_desc = [guest_desc_dict[guest] for guest in guest_list]
                    guest_desc = np.mean(guest_desc, axis=0)

    
            all_con.append(np.log10(con))
            all_T.append(T)
            all_RH.append(RH)
            all_mof_desc.append(mof_desc)
            all_guest_desc.append(guest_desc)
            all_cif.append(cif)

        all_con = torch.tensor(all_con, dtype=torch.float32)
        all_T = torch.tensor(all_T, dtype=torch.float32)
        all_RH = torch.tensor(all_RH, dtype=torch.float32)
        all_mof_desc = torch.tensor(all_mof_desc, dtype=torch.float32)
        all_guest_desc = torch.tensor(all_guest_desc, dtype=torch.float32)

        return all_con, all_T, all_RH, all_mof_desc, all_guest_desc, all_cif



    