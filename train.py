import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import cv2
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam
from glob import glob
from tensorflow.keras.metrics import Accuracy, AUC
from IPython.display import clear_output

# import different models
from unet import build_unet
from xce_resunet import build_xce_resunet
from att_xce_resunet import build_att_xce_resunet
from resunet import build_resunet
from pre_data import train_image_height, train_image_weight

# define the image size
image_height = train_image_height
image_weight = train_image_weight

# normalize the pixel in range(0,1)
def normalize(input_image):
    input_image = input_image / 255.0
    return input_image

# load image and mask
def load_data(path):
    image = sorted(glob(os.path.join(path, "image", "*.jpg")))
    mask = sorted(glob(os.path.join(path, "mask", "*.jpg")))
    return image, mask

# shuffle image and mask
def shuffling(image, mask):
    image, mask = shuffle(image, mask, random_state=19)
    return image, mask

def read_image(path):
    path = path.decode()
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    # image = cv2.resize(image, (image_weight, image_height))
    image = normalize(image)
    image = image.astype(np.float32)
    return image


def read_mask(path):
    path = path.decode()
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # mask = cv2.resize(mask, (image_weight, image_height))
    mask = normalize(mask)
    mask = mask.astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)
    return mask

def tf_parse(image, mask):
    def _parse(image, mask):
        image = read_image(image)
        mask = read_mask(mask)
        return image, mask

    image, mask = tf.numpy_function(_parse, [image, mask], [tf.float32, tf.float32])
    image.set_shape([image_weight, image_height,3])
    mask.set_shape([image_weight, image_height,1])
    return image, mask

# create dataset
def create_dataset(image, mask, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((image, mask))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(4)
    return dataset

if __name__ == "__main__":

    # Set up argparse arguments
    parser = argparse.ArgumentParser(description='Run a image segmentation algorithm.')
    parser.add_argument('batch_size', metavar='batch_size', type=int,
                        help='batch_size', default=10)
    parser.add_argument('lr', metavar='lr', type=float,
                        help='lr',default=1e-4)
    parser.add_argument('num_epochs', metavar='num_epochs', type=int,
                        help='num_epochs',default=30)
    parser.add_argument('model_type', metavar='model_type', type=int, help='model_type',default=1)

    args = parser.parse_args()
    batch_size = args.batch_size
    lr = args.lr
    num_epochs = args.num_epochs
    model_type = args.model_type

    # define path
    model_path = os.path.join("files", "model.h5")
    csv_path = os.path.join("files","data.csv")

    # load data
    dataset_path = "augmented"
    train_path = os.path.join(dataset_path, "train")
    test_path = os.path.join(dataset_path, "test")
    # train_image, train_mask), (test_image, test_mask
    train_image, train_mask = load_data(train_path)
    train_image, train_mask = shuffling(train_image, train_mask)
    test_image, test_mask = load_data(test_path)

    # print the train and test
    print(f"Train: {len(train_image)} - {len(train_mask)}")
    print(f"Train: {len(test_image)} - {len(test_mask)}")

    # create dataset
    train_dataset = create_dataset(train_image, train_mask, batch_size)
    valid_dataset = create_dataset(test_image, test_mask, batch_size)

    train_steps = len(train_image)//batch_size
    valid_steps = len(test_image)//batch_size

    if len(train_image) % batch_size != 0:
        train_steps += 1
    if len(test_image) % batch_size != 0:
        valid_steps += 1

    # load model
    if model_type == 1:
        model = build_unet((image_height, image_weight, 3))
        print("----------load unet----------")
    elif model_type == 2:
        model = build_resunet((image_height, image_weight, 3))
        print("----------load resnet----------")
    elif model_type == 3:
        model = build_xce_resunet((image_height, image_weight, 3))
        print("----------load xception-resunet----------")
    elif model_type == 4:
        model = build_att_xce_resunet((image_height, image_weight, 3))
        print("----------load attention-xception-resunet----------")
    else:
        model = build_unet((image_height, image_weight, 3))
        print("----------load unet----------")
    model.compile(loss=tf.keras.losses.binary_crossentropy,optimizer=Adam(lr), metrics=[AUC()])

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, min_lr=1e-6, verbose=1),
        CSVLogger(csv_path),
        TensorBoard(),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=False)
    ]

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks
    )

    class PlotLearning(tf.keras.callbacks.Callback):
        """
        Callback to plot the learning curves of the model during training.
        """
        def on_train_begin(self, logs={}):
            self.metrics = {}
            for metric in logs:
                self.metrics[metric] = []


        def on_epoch_end(self, epoch, logs={}):
            # Storing metrics
            for metric in logs:
                if metric in self.metrics:
                    self.metrics[metric].append(logs.get(metric))
                else:
                    self.metrics[metric] = [logs.get(metric)]

            # Plotting
            metrics = [x for x in logs if 'val' not in x]

            f, axs = plt.subplots(1, len(metrics), figsize=(15,5))
            clear_output(wait=True)

            for i, metric in enumerate(metrics):
                axs[i].plot(range(1, epoch + 2),
                            self.metrics[metric],
                            label=metric)
                if logs['val_' + metric]:
                    axs[i].plot(range(1, epoch + 2),
                                self.metrics['val_' + metric],
                                label='val_' + metric)

                axs[i].legend()
                axs[i].grid()

            plt.tight_layout()
            plt.show()


