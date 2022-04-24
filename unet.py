from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Input, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model

# convolution block
def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, kernel_size=(3,3), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, kernel_size=(3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def down_stack(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = MaxPool2D((2, 2))(x)

    return x, p

def up_stack(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, kernel_size=(2,2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)

    return x

def build_unet(input_shape):
    inputs = Input(input_shape)
    s1, p1 = down_stack(inputs, 64)
    s2, p2 = down_stack(p1, 128)
    s3, p3 = down_stack(p2, 256)
    s4, p4 = down_stack(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = up_stack(b1, s4, 512)
    d2 = up_stack(d1, s3, 256)
    d3 = up_stack(d2, s2, 128)
    d4 = up_stack(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
    model = Model(inputs, outputs, name="UNET")

    return model
