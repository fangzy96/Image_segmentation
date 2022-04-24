import numpy as np
import os
import imageio
import cv2
from albumentations import HorizontalFlip, VerticalFlip, ElasticTransform, GridDistortion, OpticalDistortion
from tqdm import tqdm
from glob import glob

train_image_height = 64
train_image_weight = 64


def load_data(path):
    # load data
    train_image = sorted(glob(os.path.join(path, 'train', 'input', '*.jpg')))
    train_mask = sorted(glob(os.path.join(path, 'train', 'target', '*.jpg')))

    test_image = sorted(glob(os.path.join(path, 'validation', 'input', '*.jpg')))
    test_mask = sorted(glob(os.path.join(path, 'validation', 'target', '*.jpg')))

    return (train_image, train_mask), (test_image, test_mask)



# data augment
def augment_data(images, masks, save_path, augment=True):
    # define image size
    image_height = train_image_height
    image_weight = train_image_weight

    for idx, (image, mask) in tqdm(enumerate(zip(images,masks)), total=len(images)):

        # for windows the split is \\, but in linux is /
        image_name = image.split("\\")[-1].split(".")[0]
        # read image and mask
        image = cv2.imread(image, cv2.IMREAD_COLOR)
        mask = imageio.imread(mask)

        # define augment methods
        if augment == True:
            # horizontal flip
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=image,mask=mask)
            image1 = augmented["image"]
            mask1 = augmented["mask"]

            # vertical flip
            aug = VerticalFlip(p=1.0)
            augmented = aug(image=image,mask=mask)
            image2 = augmented["image"]
            mask2 = augmented["mask"]

            # elastic transform
            aug = ElasticTransform(p=1, alpha=120,sigma=120*0.05,alpha_affine=120)
            augmented = aug(image=image,mask=mask)
            image3 = augmented["image"]
            mask3 = augmented["mask"]

            # grid distortion
            aug = GridDistortion(p=1)
            augmented = aug(image=image,mask=mask)
            image4 = augmented["image"]
            mask4 = augmented["mask"]

            # optical distortion
            aug = OpticalDistortion(p=1, distort_limit=2,shift_limit=0.5)
            augmented = aug(image=image,mask=mask)
            image5 = augmented["image"]
            mask5 = augmented["mask"]

            IMAGE = [image, image1, image2, image3, image4, image5]
            MASK = [mask, mask1, mask2, mask3, mask4, mask5]

        else:
            IMAGE = [image]
            MASK = [mask]

        index = 0
        for i, m in zip(IMAGE, MASK):
            # resize the image and mask
            i = cv2.resize(i, (image_weight, image_height))
            m = cv2.resize(m, (image_weight, image_height))

            if len(IMAGE) == 1:
                temp_image_name = f"{image_name}.jpg"
                temp_mask_name = f"{image_name}.jpg"
            else:
                temp_image_name = f"{image_name}_{index}.jpg"
                temp_mask_name = f"{image_name}_{index}.jpg"
            image_path = os.path.join(save_path, "image", temp_image_name)
            mask_path = os.path.join(save_path, "mask", temp_mask_name)
            # save image and mask
            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1



if __name__ == "__main__":

    # load data
    data_path = "bag_data/bag_data/"
    (train_image, train_mask), (test_image, test_mask) = load_data(data_path)

    # print the train and test
    print(f"train set: {len(train_image)}-{len(train_mask)}")
    print(f"test set: {len(test_image)}-{len(test_mask)}")

    # augment data
    augment_data(train_image,train_mask,"augmented/train/",augment=True)
    augment_data(test_image,test_mask,"augmented/test/",augment=False)

