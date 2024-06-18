# ProtonMOF

We developed a database of proton-conductive Metal-Organic Frameworks (MOFs) and applied machine learning techniques (transformer-based models) to predict their proton conductivity. 

## Installation

```python
$ conda env create -n protonmof python==3.8
$ conda activate protonmof
$ git clone https://github.com/seunghhs/ProtonMOF.git
$ cd ProtonMOF
$ pip install -e .

# download pre-trained moftransformer
$ moftransformer download pretrain_model
```

## Transfer Learning (Training)

```python
# Transfer Learning (Freeze)
$ python protonmof/train_freeze.py config/config_freeze.yml

# Transfer Learning (Unfreeze, finetuning)
$ python protonmof/train_unfreeze.py config/config_unfreeze.yml
```

## Transfer Learning (Test)

```python
# Transfer Learning (Freeze)
$ python protonmof/train_freeze.py config/config_freeze.yml --test True

# Transfer Learning (Unfreeze, finetuning)
$ python protonmof/train_unfreeze.py config/config_unfreeze.yml --test True

# If you want to use an already-trained model, edit the 'test/ckpt_path' part of the config file
$ python protonmof/train_freeze.py config/config_freeze_test.yml --test True
$ python protonmof/train_unfreeze.py config/config_unfreeze_test.yml --test True
```