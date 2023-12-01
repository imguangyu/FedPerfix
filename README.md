# [ICCV2023] FedPerfix: Towards Partial Model Personalization of Vision Transformers in Federated Learning

The code is adapted from https://github.com/TsingZ0/PFLlib

We are still working on improving the readability of the codes and removing unused commands. Please stay tuned!
## Environments
```
python=3.8.0
torch=0.13.1
cuda=11.6
```

For the dependency:
```
pip install -r requirements.txt
```

## Preparing Data

### Option 1. Create your own split

1. Change the settings in `dataset/generate_{$DATASET_NAME}.py`
2. `python generate_cifar100.py noniid - dir # for practical noniid and unbalanced scenario`

### Option 2. Use our split

Download from [Google Drive](https://drive.google.com/drive/folders/1fuH6yn4XMdzETB274Qhe0HdMTTphCcbZ?usp=sharing)

## Setup the wandb

We use [wandb](https://wandb.ai/site) to log our results. If you don't want to use it, you can remove the related codes and the results are also saved in `./logs`

## To reproduce our main results
1. Go to the `system` folder
```
cd system
```
2. Run the following commands
```
# Set the dataset and FL settings
data=cifar100
nb=100 # number of classes
nc=64 # number of clients
jr=0.125 # client sample rate
config=FedPerfix # Choose from FedAVG, Local, APFL, PerAVG, FedBN, FedBABU, FedRep, VanillaAttention, and FedPerfix.

python main.py -mtd $config -data $data -nc $nc -jr $jr -nb $nb
```

For the ablation and other results, please check the details of the code.

## For Customize settings
1. Use `--local_parts` to specific the local parts of the model
2. Use `--basic_model/-vt` to change the backbone of the model

## Citation
```
@inproceedings{sun2023fedperfix,
  title={FedPerfix: Towards Partial Model Personalization of Vision Transformers in Federated Learning},
  author={Sun, Guangyu and Mendieta, Matias and Luo, Jun and Wu, Shandong and Chen, Chen},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4988--4998},
  year={2023}
}
```


