import glob
import os
from os.path import join as pjoin
from typing import Optional, Tuple, Union

import kornia
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VOCDetection
from torchvision import transforms
from .utils import read_image







class ExplainabilityDataModule(pl.LightningDataModule):

    def __init__(self, root_path: str,dataset,
                 train_resize_size: Optional[Union[int, Tuple[int, int]]] = None,
                 eval_resize_size: Optional[Union[int, Tuple[int, int]]] = None,
                 num_workers: int = -1, batch_size: int = 32):
        """
        Initialization of ExplainabilityDataModule

        Args:
            root_path (str): path to root directory of dataset
            dataset (Dataset): Dataset
            train_resize_size (Optional[Union[int, Tuple[int, int]]], optional): resize of train image. Defaults to None.
            eval_resize_size (Optional[Union[int, Tuple[int, int]]], optional): resize of eval image. Defaults to None.
            num_workers (int, optional): number of workers. Defaults to -1.
            batch_size (int, optional): Batch size. Defaults to 32.
        """
        super(ExplainabilityDataModule, self).__init__()
        self.root_path = root_path
        self.train_resize_size = train_resize_size
        self.eval_resize_size = eval_resize_size

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset = dataset

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Function to setup validation train and test dataset from init parameters.

        Args:
            stage (Optional[str], optional): stage. Defaults to None.
        """
        self._train_ds = self.dataset(self.root_path, 'train', self.train_resize_size)
        self._val_ds = self.dataset(self.root_path, 'val', self.eval_resize_size)
        try:
            self._test_ds = self.dataset(self.root_path, 'test', self.eval_resize_size)
        except:
            pass
    def train_dataloader(self) -> DataLoader:
        """
        Create train Dataloader

        Returns:
            DataLoader: train dataloader
        """
        return DataLoader(
            self._train_ds, num_workers=self.num_workers, batch_size=self.batch_size,
            shuffle=True, drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create validation Dataloader

        Returns:
            DataLoader: validation dataloader
        """
        return DataLoader(
            self._val_ds, num_workers=self.num_workers, batch_size=self.batch_size,
            shuffle=False, drop_last=False
        )

    def test_dataloader(self) -> DataLoader:
        """
        Create test dataloader

        Returns:
            DataLoader: test dataloader
        """
        return DataLoader(
            self._test_ds, num_workers=self.num_workers, batch_size=self.batch_size,
            shuffle=False, drop_last=False
        )
