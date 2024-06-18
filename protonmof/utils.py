from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn as nn
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.distributions.normal import Normal
from typing import List, Union, Optional



def split_by_cif(data,  test_size=0.2, random_state=40):
    """
    split data to train, validation, and test, where each validation and test set includes only MOFs that did not appear in the train set.
    """
    train_idx,valid_idx, test_idx = [], [], []
    cif = sorted(list(set(data['Name'])))
    train_cif, test_cif = train_test_split(cif,test_size=test_size, random_state=random_state)

    for i, row in data.iterrows():
        if row['Name'] in train_cif:
            train_idx.append(i)
        else:
            test_idx.append(i)

    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]
    return train_data,  test_data



def split_by_cif_tvt(data, valid_size=0.2, test_size=0.2, random_state=40):
    """
    split data to train, validation, and test, where each validation and test set includes only MOFs that did not appear in the train set.
    """
    train_idx,valid_idx, test_idx = [], [], []
    cif = sorted(list(set(data['Name'])))
    residue_cif, test_cif = train_test_split(cif,test_size=test_size, random_state=random_state)
    train_cif, valid_cif = train_test_split(residue_cif, test_size=valid_size, random_state = random_state)

    for i, row in data.iterrows():
        if row['Name'] in train_cif:
            train_idx.append(i)
        elif row['Name'] in valid_cif:
            valid_idx.append(i)
        else:
            test_idx.append(i)

    train_data = data.iloc[train_idx]
    valid_data = data.iloc[valid_idx]
    test_data = data.iloc[test_idx]
    return train_data, valid_data, test_data


def make_train_test(data, train_cif, test_cif):
    train_idx, test_idx = [], []

    for i, row in data.iterrows():
        if row['Name'] in train_cif:
            train_idx.append(i)
        elif row['Name'] in test_cif:
            test_idx.append(i)
        

    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]
    return train_data, test_data

def convert_none_type(x):
    x = None if x == 'None' else x
    return x
    
def calculate_con(output, batch, arr):
    """
    calculate conductivity (log) from outputs using the Arrhenius or VTF equation.
    """

    logA = output[:,0]
    if arr == 'arrhenius':
        temp = batch['T'].reshape(-1)
        Ea = output[:,1]
        con  = arrhenius_con(logA, Ea, temp)

    elif arr == 'vtf':
        temp = batch['T'].reshape(-1)
        Ea = output[:,1]
        T0 = output[:,2]
        con  = arrhenius_con(logA, Ea, temp, T0)       

    else:
        con = output.reshape(-1)

    return con


def arrhenius_con(logA, Ea, temp):
    """
    calculate Arrhenius conductivity (log) from the Arrhenius equation.
    """
    R = 8.6173303e-5
    e = np.exp(1)
    C = np.log10(e)/R    
    con = logA - C*Ea/temp
    return con

def vtf_con(logA, Ea, temp, T0):
    """
    calculate vtf conductivity (log) from Vogel-Fulcher-Tammann (VFT) equation. 
    """
    R = 8.6173303e-5
    e = np.exp(1)
    C = np.log10(e)/R    
    con = logA - C*Ea/(temp-T0)    
    return con



def arrhenius_reg(output, arr, reg_rate = 0, slope = 0.0289, intercept=-0.0272): #slope = 0.0294, intercept=0.0991
    """
    adds arrhenius regularization term to loss function
    """
    if arr not in ['arrhenius', 'vtf']:
        return 0
    logA = output[:,0]
    Ea = output[:,1]
    
    slope = slope
    intercept = intercept
    intercept_range = 0.2

    #predict Ea from logA vs Ea regression fit equation
    pred_Ea = logA * slope + intercept
    residuals = torch.abs(pred_Ea - Ea) - intercept_range
    residuals = torch.max(residuals,torch.zeros(residuals.shape).to(residuals.device))

    return reg_rate * torch.sum(residuals)



class Normalizer(object):
    """
    normalize for regression
    """

    def __init__(self, mean, std):
        if mean and std:
            self._norm_func = lambda tensor: (tensor - mean) / std
            self._denorm_func = lambda tensor: tensor * std + mean
        else:
            self._norm_func = lambda tensor: tensor
            self._denorm_func = lambda tensor: tensor

        self.mean = mean
        self.std = std

    def encode(self, tensor):
        return self._norm_func(tensor)

    def decode(self, tensor):
        return self._denorm_func(tensor)


class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.

    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where :code:`warmup_steps = warmup_epochs * steps_per_epoch`).
    Then the learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr` over the
    course of the remaining :code:`total_steps - warmup_steps` (where :code:`total_steps =
    total_epochs * steps_per_epoch`). This is roughly based on the learning rate
    schedule from `Attention is All You Need <https://arxiv.org/abs/1706.03762>`_, section 5.3.
    """
    def __init__(self,
                 optimizer: Optimizer,
                 warmup_epochs: List[Union[float, int]],
                 total_epochs: List[int],
                 steps_per_epoch: int,
                 init_lr: List[float],
                 max_lr: List[float],
                 final_lr: List[float]):
        """
        :param optimizer: A PyTorch optimizer.
        :param warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        :param total_epochs: The total number of epochs.
        :param steps_per_epoch: The number of steps (batches) per epoch.
        :param init_lr: The initial learning rate.
        :param max_lr: The maximum learning rate (achieved after :code:`warmup_epochs`).
        :param final_lr: The final learning rate (achieved after :code:`total_epochs`).
        """
        assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == len(init_lr) == \
               len(max_lr) == len(final_lr)

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        """
        Gets a list of the current learning rates.

        :return: A list of the current learning rates.
        """
        return list(self.lr)

    def step(self, current_step: int = None):
        """
        Updates the learning rate by taking a step.

        :param current_step: Optionally specify what step to set the learning rate to.
                             If None, :code:`current_step = self.current_step + 1`.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]