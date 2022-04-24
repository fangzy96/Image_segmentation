from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Input, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model
import tensorflow as tf

def conv_block(inputs, num_filters):

    x = Conv2D(num_filters, kernel_size=(3,3), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, kernel_size=(3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    residual = Conv2D(num_filters, kernel_size=(1,1), padding="same")(inputs)
    return x, residual

def down_stack(inputs, num_filters):

    x, _ = conv_block(inputs, num_filters)
    p = MaxPool2D((2, 2))(x)
    residual = Conv2D(num_filters, kernel_size=(1,1), padding="same")(inputs)

    return x, p, residual

def up_stack(inputs, skip_features, num_filters):

    x = Conv2DTranspose(num_filters, kernel_size=(2,2), strides=2, padding="same")(inputs)
    residual = Conv2D(num_filters, kernel_size=(1,1), strides=1, padding="same")(x)
    x = Concatenate()([x, skip_features])
    x, _ = conv_block(x, num_filters)

    return x, residual

def build_resunet(input_shape):
    inputs = Input(input_shape)
    s1, p1, r1 = down_stack(inputs, 64)
    s1 = tf.add(s1, r1)
    s2, p2, r2 = down_stack(p1, 128)
    s2 = tf.add(s2, r2)
    s3, p3, r3 = down_stack(p2, 256)
    s3 = tf.add(s3, r3)
    s4, p4, r4 = down_stack(p3, 512)
    s4 = tf.add(s4, r4)

    b1, rb = conv_block(p4, 1024)
    b1 = tf.add(b1, rb)

    d1, rd1 = up_stack(b1, s4, 512)
    d1 = tf.add(d1, rd1)
    d2, rd2 = up_stack(d1, s3, 256)
    d2 = tf.add(d2, rd2)
    d3, rd3 = up_stack(d2, s2, 128)
    d3 = tf.add(d3, rd3)
    d4, rd4 = up_stack(d3, s1, 64)
    d4 = tf.add(d4, rd4)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
    model = Model(inputs, outputs, name="UNET")

    return model
