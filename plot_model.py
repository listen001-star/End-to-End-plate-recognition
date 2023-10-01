import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same'):
    x = layers.Conv2D(nb_filter, kernel_size, strides=strides, padding=padding)(x)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    return x


def Conv2dT_BN(x, filters, kernel_size, strides=(2, 2), padding='same'):
    x = layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    return x

height = 80
width = 240

inpt = layers.Input(shape=(height, width, 3))
conv1 = Conv2d_BN(inpt, 8, (3, 3))
conv1 = Conv2d_BN(conv1, 8, (3, 3))
pool1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)

conv2 = Conv2d_BN(pool1, 16, (3, 3))
conv2 = Conv2d_BN(conv2, 16, (3, 3))
pool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

conv3 = Conv2d_BN(pool2, 32, (3, 3))
conv3 = Conv2d_BN(conv3, 32, (3, 3))
pool3 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)

conv4 = Conv2d_BN(pool3, 64, (3, 3))
conv4 = Conv2d_BN(conv4, 64, (3, 3))
pool4 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4)

conv5 = Conv2d_BN(pool4, 128, (3, 3))
conv5 = layers.Dropout(0.5)(conv5)
conv5 = Conv2d_BN(conv5, 128, (3, 3))
conv5 = layers.Dropout(0.5)(conv5)

convt1 = Conv2dT_BN(conv5, 64, (3, 3))
concat1 = layers.concatenate([conv4, convt1], axis=3)
concat1 = layers.Dropout(0.5)(concat1)
conv6 = Conv2d_BN(concat1, 64, (3, 3))
conv6 = Conv2d_BN(conv6, 64, (3, 3))

convt2 = Conv2dT_BN(conv6, 32, (3, 3))
concat2 = layers.concatenate([conv3, convt2], axis=3)
concat2 = layers.Dropout(0.5)(concat2)
conv7 = Conv2d_BN(concat2, 32, (3, 3))
conv7 = Conv2d_BN(conv7, 32, (3, 3))

convt3 = Conv2dT_BN(conv7, 16, (3, 3))
concat3 = layers.concatenate([conv2, convt3], axis=3)
concat3 = layers.Dropout(0.5)(concat3)
conv8 = Conv2d_BN(concat3, 16, (3, 3))
conv8 = Conv2d_BN(conv8, 16, (3, 3))

convt4 = Conv2dT_BN(conv8, 8, (3, 3))
concat4 = layers.concatenate([conv1, convt4], axis=3)
concat4 = layers.Dropout(0.5)(concat4)
conv9 = Conv2d_BN(concat4, 8, (3, 3))
conv9 = Conv2d_BN(conv9, 8, (3, 3))
conv9 = layers.Dropout(0.5)(conv9)
outpt = layers.Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv9)

# 创建模型
model = tf.keras.Model(inputs=inpt, outputs=outpt)

# 保存模型结构图
tf.keras.utils.plot_model(model, to_file='model1.png', show_shapes=True, show_layer_names=True, dpi=96)