"""Config for resnet module for multilabel Pvoc classification """
from dataclasses import dataclass, field
import torch.nn as nn
import kornia.augmentation as KA
from src.metrics.metrics import *

from src.datasets.Pvoc import PvocAttentionDataset
from datetime import datetime

class BCEWLossConverted:
    def __call__(self, output, target):
        loss = nn.BCELoss()(output, target.to(torch.float32))
        return loss


@dataclass
class AdvPvocMC:
    """Config for adversarial ResNet-50 training on PascalVOC multilabel classification task
    for detailed values description refer to robustness library documentation:
    https://robustness.readthedocs.io/en/latest/example_usage/training_lib_part_1.html
    """

    arch: str = 'resnet50'    #chosen architecture
    dataset: str = 'pvoc'
    # pascalVoC dataset path
    data: str = '/home/User/Documents/robust-models-transfer/data'
    # log dir path
    out_dir: str = '/home/User/Documents/robust-models-transfer/out_dir'
    # task name
    exp_name: str = 'exp'
    epochs: int = 150
    lr: float = 0.005
    step_lr: int = 30
    batch_size: int = 8
    weight_decay: float = 5e-4
    adv_train: bool = True

    # optional path to pretrained model
    model_path: str = '/home/User/Downloads/resnet50_linf_eps4.0.ckpt'
    freeze_level: int = -1
    log_iters: int = 1
    momentum: float = 0.9

    constraint: str = 'inf'
    eps: float = 4.0
    attack_lr: float = 5e-6  # 2
    attack_steps: int = 3

    augmentation: KA.AugmentationSequential = KA.AugmentationSequential(
        KA.RandomEqualize(p=0.2),
        KA.RandomSharpness(p=0.2),
        KA.RandomSolarize(p=0.2),
        KA.RandomGaussianNoise(p=0.5, mean=0., std=0.05),
        KA.RandomPerspective(distortion_scale=0.5, p=0.3),
        KA.RandomElasticTransform(p=0.2),
        KA.RandomCrop((224, 224), p=1.0)
    )

    dataset_eval = PvocAttentionDataset(
        root_path='/home/User/Downloads/ML-Interpretability-Evaluation-Benchmark-master', resize_size=224)
    experiment_name = '{}'.format(
        str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    )
    normalization = KA.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
