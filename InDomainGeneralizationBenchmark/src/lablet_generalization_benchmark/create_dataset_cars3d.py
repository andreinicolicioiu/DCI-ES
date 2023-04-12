import numpy as np
from sklearn.model_selection import train_test_split
import pdb

file = '/home/anicolicioiu/data/cars3d-x64.npz'

data = np.load(file)

[print(k) for k in data.keys()]
pdb.set_trace()

# data['imgs'].shape: (17568, 64, 64, 3)
# data['factors'].shape: (17568, 3)
# data['factor_sizes']: array([  4,  24, 183])
# data['factor_names']: array(['elevation', 'azimuth', 'object'], dtype='<U9')
images_all = data['imgs']
targets_all = data['factors']

images_train, images_test, targets_train, targets_test = train_test_split(
    images_all, targets_all,
    test_size=0.2, random_state=42)

images_train, images_val, targets_train, targets_val = train_test_split(
    images_train, targets_train,
    test_size=0.1, random_state=43)

test_split_name_images = '/home/anicolicioiu/data/disent_indommain/cars3d_random_test_images.npz'
test_split_name_targets = '/home/anicolicioiu/data/disent_indommain/cars3d_random_test_labels.npz'

val_split_name_images = '/home/anicolicioiu/data/disent_indommain/cars3d_random_val_images.npz'
val_split_name_targets = '/home/anicolicioiu/data/disent_indommain/cars3d_random_val_labels.npz'

train_split_name_images = '/home/anicolicioiu/data/disent_indommain/cars3d_random_train_without_val_images.npz'
train_split_name_targets = '/home/anicolicioiu/data/disent_indommain/cars3d_random_train_without_val_labels.npz'

np.savez_compressed(test_split_name_images, images_test)
np.savez_compressed(test_split_name_targets, targets_test)

np.savez_compressed(val_split_name_images, images_val)
np.savez_compressed(val_split_name_targets, targets_val)

np.savez_compressed(train_split_name_images, images_train)
np.savez_compressed(train_split_name_targets, targets_train)