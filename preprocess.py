# resize and rescale images for preprocessing

import glob
import SimpleITK as sitk
import numpy as np
from skimage.transform import resize
import os
import multiprocessing as mp
import imgaug as ia
from imgaug import augmenters as iaa  

### Configs
# 8 cores are used for multi-thread processing
# NUM_JOBS = 8
NUM_JOBS=8
#  resized output size, can be 128 or 256
# IMG_SIZE = 256
IMG_SIZE = 128
INPUT_DATA_DIR = 'Images/'
OUTPUT_DATA_DIR = 'Results/'
# the intensity range is clipped with the two thresholds, this default is used for our CT images, please adapt to your own dataset
# LOW_THRESHOLD = -1024
# HIGH_THRESHOLD = 600
LOW_THRESHOLD = -800
HIGH_THRESHOLD = 300
# suffix (ext.) of input images
SUFFIX = '.jpeg'
# whether or not to trim blank axial slices, recommend to set as True
TRIM_BLANK_SLICES = True



# Augmentation Settings
AUGMENTATIONS = iaa.SomeOf((1, 3), [  # Apply 1 to 3 of the following augmentations:
    iaa.Fliplr(0.5),  # Horizontal flip 50% of the time
    iaa.Affine(rotate=(-20, 20)),  # Random rotation
    iaa.Multiply((0.8, 1.2)),  # Random brightness change
    iaa.GaussianBlur(sigma=(0.0, 0.5))  # Random blur
])

def resize_img(img):
    nan_mask = np.isnan(img) # Remove NaN
    if img.ndim == 2:
        img = np.stack((img,)*3, axis=-1) 
    # print(img.shape)
    img[nan_mask] = LOW_THRESHOLD
    img = np.interp(img, [LOW_THRESHOLD, HIGH_THRESHOLD], [-1,1])

    if TRIM_BLANK_SLICES:
        valid_plane_i = np.mean(img, (1,2)) != -1 # Remove blank axial planes
        img = img[valid_plane_i,:,:]

    img = resize(img, (IMG_SIZE, IMG_SIZE, IMG_SIZE), mode='constant', cval=-1)
    return img

def main():
    img_list = list(glob.glob(INPUT_DATA_DIR+"*"+SUFFIX))

    processes = []
    for i in range(NUM_JOBS):
        processes.append(mp.Process(target=batch_resize, args=(i, img_list)))
    for p in processes:
        p.start()

def batch_resize(batch_idx, img_list):
    for idx in range(len(img_list)):
        if idx % NUM_JOBS != batch_idx:
            continue
        imgname = img_list[idx].split('/')[-1]
        # print(OUTPUT_DATA_DIR+imgname.split('.')[0]+".npy")
        if os.path.exists(OUTPUT_DATA_DIR+imgname.split('.')[0]+".npy"):
            # skip images that already finished pre-processing
            continue
        try:
            img = sitk.ReadImage(img_list[idx])
        except Exception as e: 
            # skip corrupted images
            print(e)
            print("Image loading error:", imgname)
            continue 
        img = sitk.GetArrayFromImage(img)
        for _ in range(500):  # Create 3 augmented versions
            img_aug = AUGMENTATIONS.augment_image(img)
            img_aug = resize_img(img_aug)  # Resize the augmented image
            print(_)
            # Save augmented image with unique identifier
            save_name = imgname.split('.')[0] + '_aug_{}.npy'.format(_) 
            np.save(OUTPUT_DATA_DIR + save_name, img_aug)

        try:
            img = resize_img(img)
        except Exception as e: # Some images are corrupted
            print(e)
            print("Image resize error:", imgname)
            continue
        # preprocessed images are saved in numpy arrays
        np.save(OUTPUT_DATA_DIR+imgname.split('.')[0]+".npy", img)

if __name__ == '__main__':
    main()
