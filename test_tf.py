# note: keras >=2.4.0 and tensorflow >=2 required.
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from loaddata import load_cifar, load_mnist
from art.classifiers import KerasClassifier
from tests.utils import get_image_classifier_tf, get_image_classifier_kr

# load mnist model
classifier, _ = get_image_classifier_tf()
#classifier = get_image_classifier_kr()

### Create a sampling layer ###############
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

latent_dim = 2

### Build the encoder ############
encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

### Build the decoder #############
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

### Define the VAE as a Model with a custom train_step #####
class DefenseVAE(keras.Model):
    def __init__(self, encoder, decoder, classifier, **kwargs):
        super(DefenseVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier

    def train_step(self, data):
        x_adv, xy= data
        x, y = xy

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = encoder(x_adv)
            # reconstruction
            reconstruction = decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(x, reconstruction)
            )
            reconstruction_loss *= 28 * 28
            # KL loss
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            # model prediction
            y_pred = classifier.predict(reconstruction)
            model_loss = tf.reduce_mean(
                keras.losses.categorical_crossentropy(y, y_pred)
            )
            
            total_loss = reconstruction_loss + kl_loss + model_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "model_loss": model_loss,
        }

### Train the DefenseVAE ##########
# load mnist data
x_train, y_train, x_test, y_test = load_mnist()
x_train_adv = x_train.copy()

#preds = classifier.predict(x_train)
vae = DefenseVAE(encoder, decoder, classifier)
vae.compile(optimizer=keras.optimizers.Adam())
vae.run_eagerly = True
vae.fit(x_train_adv, (x_train, y_train), epochs=30, batch_size=128)

