from keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import glob
import numpy as np
import copy
import sys

cv_img = []
count = 0
for img in glob.glob(".\images/tiger\*.jpg"):
    n= cv2.imread(img)
    cv_img.append(n)
    count += 1
    if count > 149:
        break

print('Length Images:', len(cv_img))
# cv_img = np.array(cv_img)
# cv_img = np.reshape(cv_img, (len(cv_img), 224, 224, 3))


def flip_horizontally(image):
    """This function flips a given image vertically."""

    vertically_flipped_image = copy.deepcopy(image)
    center = int(len(image[0]) / 2)
    for i, row in enumerate(image):
        for j in range(center):
            vertically_flipped_image[i][j] = image[i][(len(image[0]) - 1) - j]
            vertically_flipped_image[i][(len(image[0]) - 1) - j] = image[i][j]
    return vertically_flipped_image

i = 851
for image in cv_img:
    horizontally_flipped_img = flip_horizontally(image)
    cv2.imwrite(f'{i}.jpg', horizontally_flipped_img)
    i += 1

# # I plan of having these commented lines of code in a separate file. This generates the augmented images.
# # Data Augmentation.
# datagen = ImageDataGenerator(
#             featurewise_center=False,
#             featurewise_std_normalization=False,
#             rotation_range=20,
#             width_shift_range=0.2,
#             height_shift_range=0.2,
#             horizontal_flip=True
#         )
#
# image_gen = datagen.flow(cv_img, batch_size=1, save_to_dir='./dining_table_augmented',
#                          save_prefix='image', save_format='jpg')
#
# total = 0
# for image in image_gen:
#     total += 1
#     if total == 74:
#         break