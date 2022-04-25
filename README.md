# Image_segmentation


1.pre_data.py:
It contains the image augmentation methods. In fact, based on the experiments results, we use Horizontal flip and Vertical flip.

2.train.py:
It is the train function, please run it on terminal:
python train.py batch_size learning_rate epochs model_type
For example:
python train.py 256 0.0001 20 1
Model_type: 1: UNet 2: Res-UNet 3: Xcep-Res-UNet 4: Att-Xcep-Res-UNet

3.predict.py:
It will predict the validation set

4. unet.py resunet.py xce_resunet.py att_xce_resunet.py:
They are the different models.

This project references Tensorflow tutorial: https://www.tensorflow.org/tutorials/images/segmentation?hl=en and the YouTube tutorial: https://www.youtube.com/watch?v=M3EZS__Z_XE
