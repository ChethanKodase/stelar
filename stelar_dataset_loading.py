import os
import cv2
from PIL import Image
import numpy as np
from patchify import patchify
#for  normalization of data
from sklearn.preprocessing import MinMaxScaler, StandardScaler
minmaxscaler = MinMaxScaler()


dataset_root_folder = './dataset/'
dataset_name = 'processed_lai_npy'

os.walk(dataset_root_folder)
image_patch_size = 256

image_dataset = []
mask_dataset = []
image_extension = 'jpg' # 'jpg', 'png'
image_type = 'images' # 'images', 'masks'

for image_type in ['images', 'masks']: 
    if image_type =='images':
        image_extension='jpg'
    elif image_type == 'masks':
        image_extension = 'png'
    for tile_id in range(1,8):
        for image_id in range(1, 10):
            image = cv2.imread(f'{dataset_root_folder}/{dataset_name}/Tile {tile_id}/{image_type}/image_part_00{image_id}.{image_extension}', 1)
            if image is not None:
                #print(image.shape)
                #image_dataset.append(image)
                size_x = (image.shape[1]//image_patch_size)*image_patch_size
                size_y = (image.shape[0]//image_patch_size)*image_patch_size
                #print("(image.shape[1]//image_patch_size)", (image.shape[1]//image_patch_size))
                #print("(image.shape[0]//image_patch_size)", (image.shape[0]//image_patch_size))
                #print("{} -- {} - {}".format(image.shape, size_x, size_y))
                image = Image.fromarray(image)
                image = image.crop((0, 0, size_x, size_y))
                #print("({} -- {})".format(image.size[0], image.size[1]))
                image = np.array(image)
                patched_images = patchify(image, (image_patch_size, image_patch_size, 3), step= image_patch_size)
                #print(len(patched_images))
                #print("patched_images.shape", patched_images.shape)
                for i in range(patched_images.shape[0]):
                    for j in range(patched_images.shape[1]):
                        individual_patched_image = patched_images[i,j,:,:]
                        print(individual_patched_image.shape)
                        individual_patched_image = minmaxscaler.fit_transform(individual_patched_image.reshape(-1, individual_patched_image.shape[-1])).reshape(individual_patched_image.shape)
                        individual_patched_image = individual_patched_image[0]
                        print(individual_patched_image.shape)
                        if image_type =='images':
                            image_dataset.append(individual_patched_image)
                        elif image_type == 'masks':
                            mask_dataset.append(individual_patched_image)
