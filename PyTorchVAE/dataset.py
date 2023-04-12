import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile


# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self):
        pass
    
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """
    
    def _check_integrity(self) -> bool:
        return True
    
    

class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable,
                **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"        
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
        
        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, 0.0 # dummy datat to prevent breaking 

class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
#       =========================  OxfordPets Dataset  =========================
            
#         train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                               transforms.CenterCrop(self.patch_size),
# #                                               transforms.Resize(self.patch_size),
#                                               transforms.ToTensor(),
#                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
#         val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                             transforms.CenterCrop(self.patch_size),
# #                                             transforms.Resize(self.patch_size),
#                                             transforms.ToTensor(),
#                                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

#         self.train_dataset = OxfordPets(
#             self.data_dir,
#             split='train',
#             transform=train_transforms,
#         )
        
#         self.val_dataset = OxfordPets(
#             self.data_dir,
#             split='val',
#             transform=val_transforms,
#         )
        
#       =========================  CelebA Dataset  =========================
    
        train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.CenterCrop(148),
                                              transforms.Resize(self.patch_size),
                                              transforms.ToTensor(),])
        
        val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.patch_size),
                                            transforms.ToTensor(),])
        
        self.train_dataset = MyCelebA(
            self.data_dir,
            split='train',
            transform=train_transforms,
            download=False,
        )
        
        # Replace CelebA with your dataset
        self.val_dataset = MyCelebA(
            self.data_dir,
            split='test',
            transform=val_transforms,
            download=False,
        )
#       ===============================================================
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    # def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
    #     return DataLoader(
    #         self.val_dataset,
    #         batch_size=144,
    #         num_workers=self.num_workers,
    #         shuffle=True,
    #         pin_memory=self.pin_memory,
    #     )
     

from this import d
from typing import List
import os

import numpy as np
import sklearn.utils.extmath
import torch.utils.data
import torchvision.transforms
import pdb

class IndexManger(object):
    """Index mapping from features to positions of state space atoms."""

    def __init__(self, factor_sizes: List[int]):
        """Index to latent (= features) space and vice versa.
        Args:
          factor_sizes: List of integers with the number of distinct values for
            each of the factors.
        """
        self.factor_sizes = np.array(factor_sizes)
        self.num_total = np.prod(self.factor_sizes)
        self.factor_bases = self.num_total / np.cumprod(self.factor_sizes)

        self.index_to_feat = sklearn.utils.extmath.cartesian(
            [np.array(list(range(i))) for i in self.factor_sizes])


class BenchmarkDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path, dataset_name, variant, mode, standardise=False):
        super().__init__()
        images_filename = "{}_{}_{}_images.npz".format(dataset_name, variant,
                                                       mode)
        targets_filename = "{}_{}_{}_labels.npz".format(dataset_name, variant,
                                                        mode)
        self.transform = torchvision.transforms.ToTensor()
        self._factor_sizes = None
        self._factor_names = None
        if dataset_name == 'dsprites':
            self._factor_sizes = [3, 6, 40, 32, 32]
            self._factor_names = [
                'shape', 'scale', 'orientation', 'x-position', 'y-position'
            ]
        elif dataset_name == 'shapes3d':
            self._factor_sizes = [10, 10, 10, 8, 4, 15]
            self._factor_names = [
                'floor color', 'wall color', 'object color', 'object size',
                'object type', 'azimuth'
            ]
        elif dataset_name == 'mpi3d':
            self._factor_sizes = [6, 6, 2, 3, 3, 40, 40]
            self._factor_names = [
                'color', 'shape', 'size', 'height', 'bg color', 'x-axis',
                'y-axis'
            ]

        self._index_manager = IndexManger(self._factor_sizes)

        def load_data(filename):
            if not os.path.exists(filename):
                self.download_dataset(filename)
            return np.load(filename,
                           encoding='latin1',
                           allow_pickle=True)['arr_0']

        self._dataset_images = load_data(
            os.path.join(dataset_path, images_filename))
        self._dataset_targets = load_data(
            os.path.join(dataset_path, targets_filename))

        self._dataset_targets = self._dataset_targets.astype(np.float32)
        # TODO: standardise the targets???
        # if standardise:
        #     for i in range(self._dataset_targets.shape[1]):
        #         targets = self._dataset_targets[:,i]
        #         mean = targets.mean()
        #         std = targets.std()
        #         self._dataset_targets[:,i] = self._dataset_targets[:,i] - mean
        #         self._dataset_targets[:,i] = self._dataset_targets[:,i] / std

    def __len__(self):
        return len(self._dataset_targets)

    @property
    def normalized_targets(self):
        return self._targets / (np.array(self._factor_sizes) - 1)

    @property
    def _targets(self):
        return self._index_manager.index_to_feat

    def __getitem__(self, idx: int, normalize: bool = True):
        image = self._dataset_images[idx]
        targets = self._dataset_targets[idx]
        if normalize:
            targets = targets / (np.array(self._factor_sizes).astype(np.float32) - 1)
        
        if self.transform is not None:
            image = np.transpose(image, (1, 2, 0))
            image = self.transform(image)

        return image, targets

    @staticmethod
    def download_dataset(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        from urllib import request
        if 'dsprites' in file_path.lower():
            zenodo_code = '4835774'
        elif 'shapes3d' in file_path.lower():
            zenodo_code = '4898937'
        elif 'mpi3d' in file_path.lower():
            zenodo_code = '4899346'
        else:
            raise Exception('datsaet needs to be ')
        url = 'https://zenodo.org/record/{}/files/{}?download=1'.\
            format(zenodo_code, os.path.split(file_path)[-1])

        print(f'file not found locally, downloading from {url} ...')
        request.urlretrieve(url, file_path, )


def load_dataset(dataset_name: str,
                 variant='random',
                 mode='train',
                 dataset_path=None,
                 batch_size=4,
                 num_workers=0,
                 standardise=False):
    """ Returns a torch dataset loader for the requested split
    Args:
        dataset_name (str): the dataset name, can dbe either '
            shapes3d, 'dsprites' or 'mpi3d'
        variant (str): the split variant, can be either
            'none', 'random', 'composition', 'interpolation', 'extrapolation'
        mode (str): mode, can be either 'train' or 'test', default is 'train'
        dataset_path (str): path to dataset folder
        batch_size (int): batch_size, default is 4
        num_workers (int): num_workers, default = 0
    Returns:
        dataset
    """
    dataset = BenchmarkDataset(dataset_path, dataset_name, variant, mode, standardise)
    return dataset
    # return torch.utils.data.DataLoader(dataset,
    #                                    batch_size=batch_size,
    #                                    shuffle=True,
    #                                 #    drop_last=True,  
    #                                    num_workers=num_workers)




class DisentDatasets(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        # data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        test_batch_size: int = 8,
        # patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        train_dataset=None,
        val_dataset=None,
        test_dataset=None,

        # **kwargs,
    ):
        super().__init__()

        # self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

        # self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset


#       ===============================================================
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
     
     