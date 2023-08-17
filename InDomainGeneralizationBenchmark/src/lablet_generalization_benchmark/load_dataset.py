# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from this import d
from typing import List
import os

import numpy as np
import sklearn.utils.extmath
import torch.utils.data
import torchvision.transforms

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

    def __init__(self, dataset_path, dataset_name, variant, mode, standardise=False, imagenet_normalise=False,
            data_fraction=None):
        super().__init__()
        images_filename = "{}_{}_{}_images.npz".format(dataset_name, variant,
                                                       mode)
        targets_filename = "{}_{}_{}_labels.npz".format(dataset_name, variant,
                                                        mode)
        list_transforms = [torchvision.transforms.ToTensor()]
        if imagenet_normalise:
            # on dsprites we only have one channel
            if dataset_name == 'dsprites':
                normalize = torchvision.transforms.Normalize(mean=np.array([0.485, 0.456, 0.406]).mean(),
                    std=np.array([0.229, 0.224, 0.225]).mean())
            else:
                normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
            list_transforms.append(normalize)
        
        self.transform = torchvision.transforms.Compose(list_transforms)

        self._factor_sizes = None
        self._factor_names = None

        self._factor_discrete = None
        self._factor_discrete_more = None
        self.dataset_name = dataset_name

        if dataset_name == 'dsprites':
            self._factor_sizes = [3, 6, 40, 32, 32]
            self._factor_names = [
                'shape', 'scale', 'orientation', 'x-position', 'y-position'
            ]
            self._factor_discrete = [True, False, False, False, False]
            self._factor_discrete_more = [True, True, False, False, False]

        elif dataset_name == 'shapes3d':
            # color is given in hue values, so it has an order. Although, with only 10 points,
            # probably it is better to simply define them as categorical variables 
            # since the difference between hue color is to big
            self._factor_sizes = [10, 10, 10, 8, 4, 15]
            self._factor_names = [
                'floor color', 'wall color', 'object color', 'object size',
                'object type', 'azimuth'
            ]
            self._factor_discrete = [True, True, True, False, True, False]
            self._factor_discrete_more = [True, True, True, False, True, False]


        elif dataset_name == 'mpi3d':
            self._factor_sizes = [6, 6, 2, 3, 3, 40, 40]
            self._factor_names = [
                'color', 'shape', 'size', 'height', 'bg color', 'x-axis',
                'y-axis'
            ]
            self._factor_discrete = [True, True, False, False, True, False, False]
            self._factor_discrete_more = [True, True, True, True, True, False, False]

        elif dataset_name == 'cars3d':
            self._factor_sizes = [4,  24, 183]
            self._factor_names = [
                'elevation', 'azimuth', 'object'
            ]
            self._factor_discrete = [False, False, True]

            self._factor_discrete_more = [False, False, True]
        self._index_manager = IndexManger(self._factor_sizes)

        def load_data(filename):
            if not os.path.exists(filename):
                if 'val' not in filename:
                    print(f'{filename} does not exist. try to downsload ... ')
                    self.download_dataset(filename)
                else:
                    # we must create a validation set from the training set
                     # if 'train' in images_filename:
                    # split train in train and val sets
                    from sklearn.model_selection import train_test_split

                    val_split_name_images = os.path.join(dataset_path, "{}_{}_{}_images.npz".format(dataset_name, variant, 'val'))
                    val_split_name_targets = os.path.join(dataset_path, "{}_{}_{}_labels.npz".format(dataset_name, variant, 'val'))
                    
                    train_split_name_images = os.path.join(dataset_path, "{}_{}_{}_images.npz".format(dataset_name, variant, 'train_without_val'))
                    train_split_name_targets = os.path.join(dataset_path, "{}_{}_{}_labels.npz".format(dataset_name, variant, 'train_without_val'))

                    orig_train_images_filename = os.path.join(dataset_path, "{}_{}_{}_images.npz".format(dataset_name, variant, 'train'))
                    orig_train_targets_filename = os.path.join(dataset_path, "{}_{}_{}_labels.npz".format(dataset_name, variant, 'train'))


                    self._dataset_images = load_data(orig_train_images_filename)
                    self._dataset_targets = load_data(orig_train_targets_filename)

                    images_train, images_val, targets_train, targets_val = train_test_split(
                        self._dataset_images, self._dataset_targets,
                        test_size=0.05, random_state=42)

                    np.savez_compressed(val_split_name_images, images_val)
                    np.savez_compressed(val_split_name_targets, targets_val)

                    np.savez_compressed(train_split_name_images, images_train)
                    np.savez_compressed(train_split_name_targets, targets_train)

            return np.load(filename, encoding='latin1', allow_pickle=True)['arr_0'] 

        self._dataset_images = load_data(
            os.path.join(dataset_path, images_filename))
        self._dataset_targets = load_data(
            os.path.join(dataset_path, targets_filename))

        print(f'self._dataset_images:shape {self._dataset_images.shape}')
        if data_fraction is not None and data_fraction != 1.0:
            if data_fraction < 1.0:
                num_samples = int(len(self._dataset_targets) * data_fraction)
            else:
                num_samples = data_fraction
            print(f'Loading only fraction of the dataset: {num_samples} samples')
            file_perm_train = f'./internal_files/{dataset_name}_{mode}_perm.npy'

            if os.path.exists(file_perm_train):
                perm = np.load(file_perm_train)
            else:
                perm = np.random.permutation(len(self._dataset_targets))
                np.save(file_perm_train, perm)

            self._dataset_images = self._dataset_images[perm][:num_samples]
            self._dataset_targets = self._dataset_targets[perm][:num_samples]

            # hack: for small datasets, preparing the dataloader to start iterating seems to take a long time
            # thus we repreat data such that we iterate fewer times through the dataset
            standard_len = 40000
            if self._dataset_images.shape[0] < standard_len:
                print('WARNING: creating bigger dataset for faster loading')
                rep = standard_len // self._dataset_images.shape[0]
                self._dataset_images = np.tile(self._dataset_images, (rep,1,1,1))
                self._dataset_targets = np.tile(self._dataset_targets, (rep,1))

        # if 'train' in images_filename:
        #     # split train in train and val sets
        #     from sklearn.model_selection import train_test_split
        #     val_split_name_images = os.path.join(dataset_path, images_filename)
        #     val_split_name_images = val_split_name_images.replace('train','val')

        #     val_split_name_targets = os.path.join(dataset_path, targets_filename)
        #     val_split_name_targets = val_split_name_targets.replace('train','val')

        #     train_split_name_images = os.path.join(dataset_path, images_filename)
        #     train_split_name_images = train_split_name_images.replace('train','train_without_val')

        #     train_split_name_targets = os.path.join(dataset_path, targets_filename)
        #     train_split_name_targets = train_split_name_targets.replace('train','train_without_val')


        #     images_train, images_val, targets_train, targets_val = train_test_split(
        #         self._dataset_images, self._dataset_targets,
        #         test_size=0.05, random_state=42)

        #     np.savez_compressed(val_split_name_images, images_val)
        #     np.savez_compressed(val_split_name_targets, targets_val)

        #     np.savez_compressed(train_split_name_images, images_train)
        #     np.savez_compressed(train_split_name_targets, targets_train)

        self._dataset_targets = self._dataset_targets.astype(np.float32)


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
                 standardise=False,
                 imagenet_normalise=False,
                 shuffle=True,
                 data_fraction=None):
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
    dataset = BenchmarkDataset(dataset_path, dataset_name, variant, mode, standardise, imagenet_normalise,
        data_fraction=data_fraction)

    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                    #    drop_last=True,  
                                       num_workers=num_workers,
                                       pin_memory=True)




class LabelsDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path, dataset_name, variant, mode, standardise=False,
            data_fraction=None):
        super().__init__()
        images_filename = "{}_{}_{}_images.npz".format(dataset_name, variant,
                                                       mode)
        targets_filename = "{}_{}_{}_labels.npz".format(dataset_name, variant,
                                                        mode)
        self.transform = torchvision.transforms.ToTensor()
        self._factor_sizes = None
        self._factor_names = None
        self._factor_discrete = None
        self._factor_discrete_more = None
        self.dataset_name = dataset_name
        if dataset_name == 'dsprites':
            self._factor_sizes = [3, 6, 40, 32, 32]
            self._factor_names = [
                'shape', 'scale', 'orientation', 'x-position', 'y-position'
            ]
            self._factor_discrete = [True, False, False, False, False]
            self._factor_discrete_more = [True, True, False, False, False]

        elif dataset_name == 'shapes3d':
            # color is given in hue values, so it has an order. Although, with only 10 points,
            # probably it is better to simply define them as categorical variables 
            # since the difference between hue color is to bi
            self._factor_sizes = [10, 10, 10, 8, 4, 15]
            self._factor_names = [
                'floor color', 'wall color', 'object color', 'object size',
                'object type', 'azimuth'
            ]
            self._factor_discrete = [False, False, False, False, True, False]
            self._factor_discrete_more = [True, True, True, False, True, False]


        elif dataset_name == 'mpi3d':
            self._factor_sizes = [6, 6, 2, 3, 3, 40, 40]
            self._factor_names = [
                'color', 'shape', 'size', 'height', 'bg color', 'x-axis',
                'y-axis'
            ]
            self._factor_discrete = [True, True, False, False, True, False, False]
            self._factor_discrete_more = [True, True, True, True, True, False, False]

        elif dataset_name == 'cars3d':
            self._factor_sizes = [4,  24, 183]
            self._factor_names = [
                'elevation', 'azimuth', 'object'
            ]
            self._factor_discrete = [False, False, True]
            self._factor_discrete_more = [False, False, True]

        self._index_manager = IndexManger(self._factor_sizes)

        def load_data(filename):
            if not os.path.exists(filename):
                if 'val' not in filename:
                    print(f'{filename} does not exist. try to downsload ... ')
                    self.download_dataset(filename)
                else:
                    # we must create a validation set from the training set
                     # if 'train' in images_filename:
                    # split train in train and val sets
                    from sklearn.model_selection import train_test_split

                    val_split_name_images = os.path.join(dataset_path, "{}_{}_{}_images.npz".format(dataset_name, variant, 'val'))
                    val_split_name_targets = os.path.join(dataset_path, "{}_{}_{}_labels.npz".format(dataset_name, variant, 'val'))
                    
                    train_split_name_images = os.path.join(dataset_path, "{}_{}_{}_images.npz".format(dataset_name, variant, 'train_without_val'))
                    train_split_name_targets = os.path.join(dataset_path, "{}_{}_{}_labels.npz".format(dataset_name, variant, 'train_without_val'))

                    orig_train_images_filename = os.path.join(dataset_path, "{}_{}_{}_images.npz".format(dataset_name, variant, 'train'))
                    orig_train_targets_filename = os.path.join(dataset_path, "{}_{}_{}_labels.npz".format(dataset_name, variant, 'train'))


                    self._dataset_images = load_data(orig_train_images_filename)
                    self._dataset_targets = load_data(orig_train_targets_filename)

                    images_train, images_val, targets_train, targets_val = train_test_split(
                        self._dataset_images, self._dataset_targets,
                        test_size=0.05, random_state=42)

                    np.savez_compressed(val_split_name_images, images_val)
                    np.savez_compressed(val_split_name_targets, targets_val)

                    np.savez_compressed(train_split_name_images, images_train)
                    np.savez_compressed(train_split_name_targets, targets_train)

            return np.load(filename, encoding='latin1', allow_pickle=True)['arr_0'] 

        self._dataset_targets = load_data(
            os.path.join(dataset_path, targets_filename))

        if data_fraction is not None and data_fraction != 1.0:
            if data_fraction < 1.0:
                num_samples = int(len(self._dataset_targets) * data_fraction)
            else:
                num_samples = data_fraction
            print(f'Loading only fraction of the dataset: {num_samples} samples')
            
            file_perm_train = f'./internal_files/{dataset_name}_{mode}_perm.npy'

            if os.path.exists(file_perm_train):
                perm = np.load(file_perm_train)
            else:
                perm = np.random.permutation(len(self._dataset_targets))
                np.save(file_perm_train, perm)

            self._dataset_targets = self._dataset_targets[perm][:num_samples]
            
            # hack: for small datasets, preparing the dataloader to start iterating seems to take a long time
            # thus we repreat data such that we iterate fewer times through the dataset
            standard_len = 40000
            if self._dataset_targets.shape[0] < standard_len:
                print('WARNING: creating bigger dataset for faster loading')
                rep = standard_len // self._dataset_targets.shape[0]
                self._dataset_targets = np.tile(self._dataset_targets, (rep,1))

        self._dataset_targets = self._dataset_targets.astype(np.float32)

    def __len__(self):
        return len(self._dataset_targets)

    @property
    def normalized_targets(self):
        return self._targets / (np.array(self._factor_sizes) - 1)

    @property
    def _targets(self):
        return self._index_manager.index_to_feat

    def __getitem__(self, idx: int, normalize: bool = True):
        targets = self._dataset_targets[idx]
        if normalize:
            targets = targets / (np.array(self._factor_sizes).astype(np.float32) - 1)
        
        return targets, targets

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


def load_labels_dataset(dataset_name: str,
                 variant='random',
                 mode='train',
                 dataset_path=None,
                 batch_size=4,
                 num_workers=0,
                 standardise=False,
                 shuffle=True,
                 data_fraction=None):
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
    dataset = LabelsDataset(dataset_path, dataset_name, variant, mode, standardise,
        data_fraction=data_fraction)

    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                    #    drop_last=True,  
                                       num_workers=num_workers,
                                       pin_memory=True)






class RepresentationDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path, dataset_name, mode, data_fraction=None):
        super().__init__()

        self._factor_sizes = None
        self._factor_names = None
        self._factor_discrete = None
        self._factor_discrete_more = None
        self.dataset_name = dataset_name

        if dataset_name == 'dsprites':
            self._factor_sizes = [3, 6, 40, 32, 32]
            self._factor_names = [
                'shape', 'scale', 'orientation', 'x-position', 'y-position'
            ]
            self._factor_discrete = [True, False, False, False, False]
            self._factor_discrete_more = [True, True, False, False, False]

        elif dataset_name == 'shapes3d':
            # color is given in hue values, so it has an order. Although, with only 10 points,
            # probably it is better to simply define them as categorical variables 
            # since the difference between hue color is to bi
            self._factor_sizes = [10, 10, 10, 8, 4, 15]
            self._factor_names = [
                'floor color', 'wall color', 'object color', 'object size',
                'object type', 'azimuth'
            ]
            self._factor_discrete = [False, False, False, False, True, False]
            self._factor_discrete_more = [True, True, True, False, True, False]


        elif dataset_name == 'mpi3d':
            self._factor_sizes = [6, 6, 2, 3, 3, 40, 40]
            self._factor_names = [
                'color', 'shape', 'size', 'height', 'bg color', 'x-axis',
                'y-axis'
            ]
            # self._factor_discrete = [False, True, False, False, False, False, False]
            self._factor_discrete = [True, True, False, False, True, False, False]
            self._factor_discrete_more = [True, True, True, True, True, False, False]
        
        elif dataset_name == 'cars3d':
            self._factor_sizes = [4,  24, 183]
            self._factor_names = [
                'elevation', 'azimuth', 'object'
            ]
            # self._factor_discrete = [False, True, False, False, False, False, False]
            self._factor_discrete = [False, False, True]
            self._factor_discrete_more = [False, False, True]
            
        self._index_manager = IndexManger(self._factor_sizes)

        feats_filename = dataset_path + '_' + mode + '_feats.npy'
        print(f'Load features from: {feats_filename}')

        if os.path.exists(feats_filename):
            self._dataset_feats = np.load(feats_filename)
            print('Loaded')


        targets_filename = dataset_path + '_' + mode + '_targets.npy'
        print(f'Load targets from: {targets_filename}')

        if os.path.exists(targets_filename):
            self._dataset_targets = np.load(targets_filename)
            print('Loaded')
        
        if data_fraction is not None and data_fraction != 1.0:
            if data_fraction < 1.0:
                num_samples = int(len(self._dataset_targets) * data_fraction)
            else:
                num_samples = data_fraction
            print(f'Loading only fraction of the dataset: {num_samples} samples')
            
            file_perm_train = f'./internal_files/{dataset_name}_{mode}_perm.npy'

            if os.path.exists(file_perm_train):
                perm = np.load(file_perm_train)
            else:
                perm = np.random.permutation(len(self._dataset_targets))
                np.save(file_perm_train, perm)

            self._dataset_feats = self._dataset_feats[perm][:num_samples]
            self._dataset_targets = self._dataset_targets[perm][:num_samples]

            # hack: for small datasets, preparing the dataloader to start iterating seems to take a long time
            # thus we repreat data such that we iterate fewer times through the dataset
            standard_len = 40000
            if self._dataset_feats.shape[0] < standard_len:
                print('WARNING: creating bigger dataset for faster loading')
                rep = standard_len // self._dataset_feats.shape[0]
                self._dataset_feats = np.tile(self._dataset_feats, (rep,1))
                self._dataset_targets = np.tile(self._dataset_targets, (rep,1))

    def __len__(self):
        return len(self._dataset_targets)

    @property
    def normalized_targets(self):
        return self._targets / (np.array(self._factor_sizes) - 1)

    @property
    def _targets(self):
        return self._index_manager.index_to_feat

    def __getitem__(self, idx: int, normalize: bool = True):
        image = self._dataset_feats[idx]
        targets = self._dataset_targets[idx]
        return image, targets


def load_representation_dataset(dataset_name: str,
                 variant='random',
                 mode='train',
                 dataset_path=None,
                 batch_size=4,
                 num_workers=0,
                 standardise=False,
                 shuffle=True,
                 data_fraction=None):
    """ Returns a torch dataset loader for the requested split
    Args:
        dataset_name (str): the dataset name, can dbe either '
            shapes3d, 'dsprites' or 'mpi3d'
        mode (str): mode, can be either 'train' or 'test', default is 'train'
        dataset_path (str): path to dataset folder
        batch_size (int): batch_size, default is 4
        num_workers (int): num_workers, default = 0
    Returns:
        dataset
    """
    dataset = RepresentationDataset(dataset_path, dataset_name, mode, data_fraction=data_fraction)

    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                    #    drop_last=True,  
                                       num_workers=num_workers)



