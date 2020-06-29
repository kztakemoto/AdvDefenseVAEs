import numpy as np
import keras
import tensorflow as tf
from keras import backend
from loaddata import load_cifar, load_mnist
from art.attacks.evasion import FastGradientMethod
from art.config import ART_NUMPY_DTYPE
import matplotlib.pyplot as plt
from models.mnistmodel import mnist_model

from keras.layers import Flatten, Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Lambda, Reshape, Conv2DTranspose, \
    AveragePooling2D
from keras.models import Model
from keras import backend as K
from keras.losses import binary_crossentropy, mean_squared_error

from tests.utils import get_image_classifier_kr

#tf.compat.v1.enable_eager_execution()
#tf.compat.v1.disable_eager_execution()

# load pytorch mnist model available in ART
#classifier = get_image_classifier_kr()

# Mnist model
#x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
#y = tf.placeholder(tf.float32, shape=(None, 10))
mnist_model = mnist_model()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
mnist_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
mnist_model.load_weights("trained_model/mnist_model.h5")
#classifier = KerasClassifier(model=mnist_model)

def sampling(args):
    z_mean, z_log_sigma = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_sigma) * epsilon


def DefenseVAE(latent_dim=256, classifier=mnist_model):
    adv_images = Input(shape=(28, 28, 1)) #
    clean_images = Input(shape=(28, 28, 1))
    labels = Input(shape=(10,))

    latent_dim = latent_dim
    classifier = mnist_model

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(adv_images)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    if latent_dim == 128:
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

    conv_shape = x.get_shape().as_list()[1:]
    conv_dim = int(conv_shape[0]) * int(conv_shape[1]) * int(conv_shape[2])

    x = Flatten()(x)
    z_mean = Dense(latent_dim)(x)
    z_log_sigma = Dense(latent_dim)(x)

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])
    encoder = Model(adv_images, [z_mean, z_log_sigma, z], name='encoder')
    print(encoder.summary())

    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(conv_dim)(latent_inputs)
    x = Reshape(conv_shape)(x)

    if latent_dim == 128:
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    if latent_dim == 128:
        x = Conv2D(16, (3, 3), activation='relu')(x)
    else:
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    decoder = Model(latent_inputs, decoded, name='decoder')
    print(decoder.summary())
    reconstruction = decoder(encoder(adv_images)[2])
    vae = Model([adv_images, clean_images, labels], reconstruction, name='VAE')

    reconstruction_loss = K.sum(keras.losses.binary_crossentropy(clean_images, reconstruction), axis=-1)
    reconstruction_loss *= 784

    kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    print(reconstruction.shape)
    y_pred = classifier.predict(reconstruction)
    print(y_pred.shape)
    model_loss = K.sum(
        keras.losses.categorical_crossentropy(labels, y_pred),
        axis=-1
    )
    vae_loss = K.mean(reconstruction_loss + kl_loss + model_loss)
    vae.add_loss(vae_loss)

    return vae


x_train, y_train, x_test, y_test = load_mnist()
x_train_adv = x_train.copy()

model = DefenseVAE()
model.compile(optimizer='adam')

model.fit([x_train_adv, x_train, y_train], epochs=30, batch_size=128)
