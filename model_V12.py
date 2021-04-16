# -*- coding:utf-8 -*-
import tensorflow as tf

l2 = tf.keras.regularizers.l2

def modified_GEI_model(input_shape=(128, 88, 1), weight_decay=0.00001):

    h = h1 = inputs = tf.keras.Input(input_shape)

    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=7,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(h)

    h = tf.keras.layers.ZeroPadding2D((2,2))(h)
    h = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=5,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=2, padding="same")(h)

    h = tf.keras.layers.ZeroPadding2D((1,1))(h)
    h = tf.keras.layers.Conv2D(filters=256,
                               kernel_size=3,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.MaxPooling2D(pool_size=(4,4), strides=2, padding="same")(h)

    # 각 파드별로 위와같이 진행하고 마지막에 나온 feature들의 관계(작은가 큰가?)를 정의한 뒤 마지막 feature로 계산

    h1 = tf.keras.layers.ZeroPadding2D((2,2))(h1)
    h1 = tf.keras.layers.Conv2D(filters=64,
                                kernel_size=5,
                                strides=1,
                                padding="valid",
                                use_bias=False,
                                kernel_regularizer=l2(weight_decay))(h1)
    h1 = tf.keras.layers.BatchNormalization()(h1)
    h1 = tf.keras.layers.ReLU()(h1)

    h1 = tf.keras.layers.MaxPooling2D(pool_size=(4,4), strides=2, padding="same")(h1)

    h1 = tf.keras.layers.ZeroPadding2D((3,3))(h1)
    h1 = tf.keras.layers.Conv2D(filters=128,
                                kernel_size=7,
                                strides=1,
                                padding="valid",
                                use_bias=False,
                                kernel_regularizer=l2(weight_decay))(h1)
    h1 = tf.keras.layers.BatchNormalization()(h1)
    h1 = tf.keras.layers.ReLU()(h1)

    h1 = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=2, padding="same")(h1)

    h1 = tf.keras.layers.ZeroPadding2D((1,1))(h1)
    h1 = tf.keras.layers.Conv2D(filters=256,
                                kernel_size=3,
                                strides=1,
                                padding="valid",
                                use_bias=False,
                                kernel_regularizer=l2(weight_decay))(h1)
    h1 = tf.keras.layers.BatchNormalization()(h1)
    h1 = tf.keras.layers.ReLU()(h1)

    h1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(h1)

    h = (h + h1) / 2.

    h = tf.keras.layers.Conv2D(filters=1440,
                               kernel_size=1,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.GlobalMaxPool2D()(h)

    h = tf.keras.layers.Reshape((36, 40))(h)

    h2 = tf.keras.layers.Dense(9)(tf.reduce_max(h, 2))

    h3 = tf.keras.layers.Dense(10)(tf.reduce_mean(h, 1))


    return tf.keras.Model(inputs=inputs, outputs=[h2, h3])