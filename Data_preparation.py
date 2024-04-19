import os
from yaimpl.image_folder_utils import get_data_info_from_img_folder_with_splits
import glob
import pandas as pd
import random
# Make a data info csv file that includes the image paths, labels, and 
# train/val/test splits. This csv file should have three columns: ['fpath', 'label', 'split'].
metadata_dir = './metadata/'
if not os.path.exists(metadata_dir):
    os.makedirs(metadata_dir)
img_root_dir = ''
#based on the img_root_dir, the data_info csv file will be saved to the same directory as the img_root_dir

# first create the data_info csv file with fpath and label columns
image_dirs = glob.glob(os.path.join(img_root_dir, '*', '*.png')) 
labels     = [x.split('/')[-2] for x in image_dirs]

### balance the dataset
from collections import Counter
counter = Counter(labels)

# Counter({'Myeloid': 199, 'Lymphoid': 199, 'Erythroid': 199, 'Other': 199})

# get the maximum number of images per class
max_num_imgs_per_class = max(counter.values())

balanced_image_dirs = []
balanced_labels     = []    
for label in counter.keys():
    img_dirs = glob.glob(os.path.join(img_root_dir, label, '*.png')) 
    if len(img_dirs) < max_num_imgs_per_class:
        # how many more we need to add
        num_more = max_num_imgs_per_class - len(img_dirs)
        if num_more > len(img_dirs):
            # if we need more than all the images in this class, we will just duplicate the images as whole
            folds = num_more // len(img_dirs)
            remainder = num_more % len(img_dirs)
            for i in range(folds):
                balanced_image_dirs += img_dirs
                balanced_labels     += [label] * len(img_dirs)
            
            balanced_image_dirs += random.sample(img_dirs, remainder)
            balanced_labels     += [label] * remainder

            ## add the original images
            balanced_image_dirs += img_dirs
            balanced_labels     += [label] * len(img_dirs)
        else:
            # if we need less than all the images in this class, we will randomly sample the images
            balanced_image_dirs += random.sample(img_dirs, num_more)
            balanced_labels     += [label] * num_more

            ## add the original images
            balanced_image_dirs += img_dirs
            balanced_labels     += [label] * len(img_dirs)
    else:
        # if we have enough images in this class, we will randomly sample the images
        assert len(img_dirs) == max_num_imgs_per_class, 'something is wrong'
        balanced_image_dirs += img_dirs
        balanced_labels     += [label] * len(img_dirs)

# check if all the cell class are the same number
counter = Counter(balanced_labels)
# Counter({'Myeloid': 199, 'Lymphoid': 199, 'Erythroid': 199, 'Other': 199})



# get the image paths and labels for the 
data_info  = pd.DataFrame({'fpath': balanced_image_dirs, 'label': balanced_labels})

### random split, and gave the split column a value of 'train', 'val', or 'test'
data_info['split'] = 'train'
### split 20% of the data into the test set, proportionally to the original class distribution

data_info.loc[data_info.groupby('label').sample(frac=0.2).index, 'split'] = 'test'
### split 20% of the 'train' data into the validation set, proportionally to the original class distribution
data_info.loc[data_info[data_info['split']=='train'].groupby('label').sample(frac=0.2).index, 'split'] = 'val'
### save the data_info csv file
data_info.to_csv(os.path.join(metadata_dir, 'data_info.csv'), index=False)

data_info_fpath = os.path.join(metadata_dir, 'data_info.csv')