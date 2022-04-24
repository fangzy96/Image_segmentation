import tensorflow as tf
import numpy as np
import os
import pandas as pd
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import cv2
from tqdm import tqdm
from sklearn.metrics import accuracy_score, auc
from train import load_data,normalize

# define the validation image size
image_height = 256
image_weight = 256

def read_image(path):

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    # image = cv2.resize(image, (image_weight, image_height))
    ori_image = image
    image = normalize(image)
    image = image.astype(np.float32)
    return ori_image, image


def read_mask(path):

    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # mask = cv2.resize(mask, (image_weight, image_height))
    ori_mask = mask
    mask = normalize(mask)
    mask = mask.astype(np.int32)
    return ori_mask, mask

# save results
def save_results(ori_image, ori_mask, mask_pred, save_image_path):
    line = np.ones((image_height, 10, 3)) * 255
    ori_mask = np.expand_dims(ori_mask, axis=-1)
    ori_mask = np.concatenate([ori_mask, ori_mask, ori_mask], axis=-1)
    mask_pred = np.expand_dims(mask_pred, axis=-1)
    mask_pred = np.concatenate([mask_pred, mask_pred, mask_pred], axis=-1) * 255

    bag_images = np.concatenate([ori_image, line, ori_mask, line, mask_pred], axis=1)
    cv2.imwrite(save_image_path, bag_images)

if __name__ == "__main__":

    # load the moved model
    model = tf.keras.models.load_model("files/model.h5")

    # load data
    dataset_path = os.path.join("augmented", "pred")
    test_image, test_mask = load_data(dataset_path)

    # predict
    score = []
    for image, mask in tqdm(zip(test_image, test_mask), total=len(test_image)):
        image_name = image.split("\\")[-1].split(".")[0]

        ori_image, image = read_image(image) # 256 256 3
        ori_mask, mask = read_mask(mask)
        mask_pred = model.predict(np.expand_dims(image, axis=0))[0] # 1 256 256 3
        # y_show = y_pred.astype(np.int32)
        # y_show = np.squeeze(y_show, axis=-1)
        # print(y_show)
        mask_pred = mask_pred > 0.5
        # print(y_pred.shape)
        mask_pred = mask_pred.astype(np.int32)
        mask_pred = np.squeeze(mask_pred, axis=-1)
        # print(y_pred)

        # save results
        save_image_path = f"results/{image_name}.jpg"
        save_results(ori_image, ori_mask, mask_pred, save_image_path)

        mask = mask.flatten()
        mask_pred = mask_pred.flatten()

        # evaluation
        acc = accuracy_score(mask, mask_pred)
        score.append([image_name, acc])
    # print(score)
    score_res = [s[1:] for s in score]
    # print(score_res)
    score_res = np.mean(score_res, axis=0)
    print(f"Accuracy: {score_res[0]:0.5f}")
    df = pd.DataFrame(score, columns=["Image", "Acc"])
    df.to_csv("files/score.csv")

