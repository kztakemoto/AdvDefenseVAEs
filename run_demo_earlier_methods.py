import os
import numpy as np
import tensorflow as tf
import torch.nn as nn
import torch.optim as optim
from loaddata import load_cifar, load_mnist
from art.attacks.evasion import FastGradientMethod
from art.defences.preprocessor import (
    FeatureSqueezing,
    SpatialSmoothing,
    LabelSmoothing,
    GaussianAugmentation,
    TotalVarMin,
    PixelDefend,
    ThermometerEncoding,
    JpegCompression,
)
from art.defences.postprocessor import (
    HighConfidence,
    GaussianNoise,
    ClassLabels,
    Rounded,
    ReverseSigmoid,
)
from models.mnistmodel import mnist_model
from art.classifiers import KerasClassifier, PyTorchClassifier
from art.defences.preprocessor.inverse_gan import InverseGAN
from art.attacks.evasion import FastGradientMethod
from tests.utils import get_gan_inverse_gan_ft
from utils.resources.create_inverse_gan_models import build_gan_graph, build_inverse_gan_graph, load_model
from art.estimators.encoding.tensorflow import TensorFlowEncoder
from art.estimators.generation.tensorflow import TensorFlowGenerator
import logging
import matplotlib.pyplot as plt

# Configure a logger to capture outputs; these are printed in console and the level of detail is set to INFO
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# load mnist data
x_train, y_train, x_test, y_test = load_mnist()
x_test = x_test[0:1000]
y_test = y_test[0:1000]

# load mnist CNN model in Keras
logger.info('MNIST Dataset')
# Mnist model
x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
y = tf.placeholder(tf.float32, shape=(None, 10))
mnist_model, logits = mnist_model(input_ph=x, logits=True)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
mnist_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
mnist_model.load_weights("trained_model/mnist_model.h5")
classifier = KerasClassifier(model=mnist_model)

# generate adversarial images using FGSM
attack = FastGradientMethod(classifier, eps=0.13)
X_adv = attack.generate(x_test)
X_adv = np.clip(X_adv, 0, 1)

# accuracy
preds_x_test = np.argmax(classifier.predict(x_test), axis=1)
acc = np.sum(preds_x_test == np.argmax(y_test, axis=1)) / y_test.shape[0]
logger.info('Accuracy on clean test images: %.2f%%', (acc * 100))
# fooling rate
probs_X_adv = classifier.predict(X_adv)
preds_X_adv = np.argmax(probs_X_adv, axis=1)
fooling_rate = np.sum(preds_X_adv != np.argmax(y_test, axis=1)) / y_test.shape[0]
logger.info('Fooling rate of FGSM attacks: %.2f%%', (fooling_rate  * 100))

# clip
def norm(x):
    return np.clip(x, 0, 1)

# plot images
def img_plot(labels, preds_clean, preds_adv, preds_def, X_clean, X_adv, X_def, method_type):
    X_clean = X_clean.reshape(X_clean.shape[:3])
    X_adv = X_adv.reshape(X_adv.shape[:3])
    X_def = X_def.reshape(X_def.shape[:3])
    # set MNIST labels
    label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # select test images
    idx = (preds_clean == np.argmax(labels, axis=1)).reshape(1,len(labels))
    tmp_idx = preds_adv != np.argmax(labels, axis=1)
    idx = np.vstack([idx, tmp_idx])
    tmp_idx = preds_def == np.argmax(labels, axis=1)
    idx = np.vstack([idx, tmp_idx])
    idx_sample_imgs = np.sum(idx, axis=0) == 3
    idx_sample_imgs = np.array(range(len(labels)))[idx_sample_imgs].tolist()
    idx_sample_imgs = np.random.choice(idx_sample_imgs, 3, replace=False).tolist()
    fig, ax = plt.subplots(len(idx_sample_imgs), 3)
    for i, idx_img in enumerate(idx_sample_imgs):
        ax[i][0].imshow(norm(X_clean[idx_img]))
        ax[i][0].axis('off')
        ax[i][0].set_title(label[np.argmax(labels[idx_img])])
        ax[i][1].imshow(norm(X_adv[idx_img]))
        ax[i][1].axis('off')
        ax[i][1].set_title(label[preds_adv[idx_img]])
        ax[i][2].imshow(norm(X_def[idx_img]))
        ax[i][2].axis('off')
        ax[i][2].set_title(label[preds_def[idx_img]])
    plt.savefig('assets/plot_mnist_' + method_type + '.png')

#### Adversarial defenses ###########
### PREPROCESS ###################
# Feature Squeezing https://arxiv.org/abs/1704.01155
preproc = FeatureSqueezing(clip_values=(0, 1), bit_depth=1)
X_def, _ = preproc(X_adv)
preds_X_def = np.argmax(classifier.predict(X_def), axis=1)
fooling_rate = np.sum(preds_X_def != np.argmax(y_test, axis=1)) / y_test.shape[0]
logger.info('Fooling rate after Feature Squeezing: %.2f%%', (fooling_rate  * 100))
img_plot(y_test, preds_x_test, preds_X_adv, preds_X_def, x_test, X_adv, X_def, "feature_squeezing")

# Spatial Smoothing https://arxiv.org/abs/1704.01155
spatial_smoothing = SpatialSmoothing(window_size=4)
X_def, _ = spatial_smoothing(X_adv)
preds_X_def = np.argmax(classifier.predict(X_def), axis=1)
fooling_rate = np.sum(preds_X_def != np.argmax(y_test, axis=1)) / y_test.shape[0]
logger.info('Fooling rate after Spatial Smoothing: %.2f%%', (fooling_rate  * 100))
img_plot(y_test, preds_x_test, preds_X_adv, preds_X_def, x_test, X_adv, X_def, "spatial_smoothing")

# Label Smoothing https://pdfs.semanticscholar.org/b5ec/486044c6218dd41b17d8bba502b32a12b91a.pdf
ls = LabelSmoothing(max_value=0.5)
preds_X_adv = np.argmax(classifier.predict(X_adv), axis=1)
_, y_test_smooth = ls(None, y_test)
fooling_rate = np.sum(preds_X_adv != np.argmax(y_test, axis=1)) / y_test.shape[0]
logger.info('Fooling rate after Label Smoothing: %.2f%%', (fooling_rate  * 100))

# Total Variance Minimization https://arxiv.org/abs/1711.00117
preproc = TotalVarMin(clip_values=(0,1))
X_def, _ = preproc(X_adv)
preds_X_def = np.argmax(classifier.predict(X_def), axis=1)
fooling_rate = np.sum(preds_X_def != np.argmax(y_test, axis=1)) / y_test.shape[0]
logger.info('Fooling rate after Variance Minimization: %.2f%%', (fooling_rate  * 100))
img_plot(y_test, preds_x_test, preds_X_adv, preds_X_def, x_test, X_adv, X_def, "variance_minimization")

# Thermometer Encoding https://openreview.net/forum?id=S18Su--CW
preproc = ThermometerEncoding(clip_values=(0, 1), num_space=4) # devided into 4 levels
X_def, _ = preproc(X_adv)
preds_X_def = np.argmax(classifier.predict(X_def[:,:,:,1].reshape(1000,28,28,1)), axis=1) # use 2nd level
fooling_rate = np.sum(preds_X_def != np.argmax(y_test, axis=1)) / y_test.shape[0]
logger.info('Fooling rate after Thermometer Encoding: %.2f%%', (fooling_rate  * 100))
img_plot(y_test, preds_x_test, preds_X_adv, preds_X_def, x_test, X_adv, X_def[:,:,:,1].reshape(1000,28,28,1), "thermometer_encoding")

# Pixel Defend (simple PixelCNN)
class ModelImage(nn.Module):
    def __init__(self):
        super(ModelImage, self).__init__()
        self.fc = nn.Linear(28 * 28 * 1, 28 * 28 * 1 * 256)
    def forward(self, x):
        x = x.view(-1, 28 * 28 * 1)
        logit_output = self.fc(x)
        logit_output = logit_output.view(-1, 28, 28, 1, 256)
        return logit_output

model = ModelImage()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
pixelcnn = PyTorchClassifier(
    model=model, loss=loss_fn, optimizer=optimizer, input_shape=(1, 28, 28), nb_classes=10, clip_values=(0, 1)
)
preproc = PixelDefend(eps=5, pixel_cnn=pixelcnn)
X_def, _ = preproc(X_adv)
preds_X_def = np.argmax(classifier.predict(X_def), axis=1)
fooling_rate = np.sum(preds_X_def != np.argmax(y_test, axis=1)) / y_test.shape[0]
logger.info('Fooling rate after Pixel Defend: %.2f%%', (fooling_rate  * 100))
img_plot(y_test, preds_x_test, preds_X_adv, preds_X_def, x_test, X_adv, X_def, "pixel_defend")

# JPEG compression https://arxiv.org/abs/1608.00853
preproc = JpegCompression(clip_values=(0, 1))
X_def, _ = preproc(X_adv)
preds_X_def = np.argmax(classifier.predict(X_def), axis=1)
fooling_rate = np.sum(preds_X_def != np.argmax(y_test, axis=1)) / y_test.shape[0]
logger.info('Fooling rate after Jpeg Compression: %.2f%%', (fooling_rate  * 100))
img_plot(y_test, preds_x_test, preds_X_adv, preds_X_def, x_test, X_adv, X_def, "JPEG_compression")

# Inverse GAN https://arxiv.org/abs/1911.10291
# note: Inverse GAN not trained well.
# run adversarial-robustness-toolbox/utils/resources/create_inverse_gan_models.py
# for obtaining (training) Inverse GAN model.
sess = tf.Session()
gen_tf, enc_tf, z_ph, image_to_enc_ph = load_model(sess, "model-dcgan", "/Users/takemoto/inverseGAN/")
gan = TensorFlowGenerator(input_ph=z_ph, model=gen_tf, sess=sess,)
inverse_gan = TensorFlowEncoder(input_ph=image_to_enc_ph, model=enc_tf, sess=sess,)
preproc = InverseGAN(sess=sess, gan=gan, inverse_gan=inverse_gan)
#preproc = InverseGAN(sess=sess, gan=gan, inverse_gan=None) # DefenseGAN
X_def = preproc(X_adv, maxiter=20)
preds_X_def = np.argmax(classifier.predict(X_def), axis=1)
fooling_rate = np.sum(preds_X_def != np.argmax(y_test, axis=1)) / y_test.shape[0]
logger.info('Fooling rate after Inverse GAN: %.2f%%', (fooling_rate  * 100))
img_plot(y_test, preds_x_test, preds_X_adv, preds_X_def, x_test, X_adv, X_def, "inverse_GAN")

### POSTPROCESS ###################
# High Confidence
postproc = HighConfidence(cutoff=0.1)
post_probs = postproc(preds=probs_X_adv)
fooling_rate = np.sum(np.argmax(post_probs, axis=1) != np.argmax(y_test, axis=1)) / y_test.shape[0]
logger.info('Fooling rate after High Confidence: %.2f%%', (fooling_rate  * 100))

# Gaussian Noise
postproc = GaussianNoise(scale=0.1)
post_probs = postproc(preds=probs_X_adv[0:1])
for i in range(1, len(X_adv)):
    tmp_post_probs = postproc(preds=probs_X_adv[i:i+1])
    post_probs = np.vstack((post_probs, tmp_post_probs))

fooling_rate = np.sum(np.argmax(post_probs, axis=1) != np.argmax(y_test, axis=1)) / y_test.shape[0]
logger.info('Fooling rate after Gaussian Noise: %.2f%%', (fooling_rate  * 100))

# Class Labels
postproc = ClassLabels()
post_probs = postproc(preds=probs_X_adv[0:1])
for i in range(1, len(X_adv)):
    tmp_post_probs = postproc(preds=probs_X_adv[i:i+1])
    post_probs = np.vstack((post_probs, tmp_post_probs))

fooling_rate = np.sum(np.argmax(post_probs, axis=1) != np.argmax(y_test, axis=1)) / y_test.shape[0]
logger.info('Fooling rate after Class Labels: %.2f%%', (fooling_rate  * 100))

# Rounded
postproc = Rounded(decimals=2)
post_probs = postproc(preds=probs_X_adv[0:1])
for i in range(1, len(X_adv)):
    tmp_post_probs = postproc(preds=probs_X_adv[i:i+1])
    post_probs = np.vstack((post_probs, tmp_post_probs))

fooling_rate = np.sum(np.argmax(post_probs, axis=1) != np.argmax(y_test, axis=1)) / y_test.shape[0]
logger.info('Fooling rate after Class Labels: %.2f%%', (fooling_rate  * 100))

# Reverse Sigmoid
postproc = ReverseSigmoid(beta=0.5, gamma=0.1)
post_probs = postproc(preds=probs_X_adv[0:1])
for i in range(1, len(X_adv)):
    tmp_post_probs = postproc(preds=probs_X_adv[i:i+1])
    post_probs = np.vstack((post_probs, tmp_post_probs))

fooling_rate = np.sum(np.argmax(post_probs, axis=1) != np.argmax(y_test, axis=1)) / y_test.shape[0]
logger.info('Fooling rate after Reverse Sigmoid: %.2f%%', (fooling_rate  * 100))

"""
--- Memo ---
InverseGAN (An Lin et al. 2019) # only used in tensorflow
DefenseGAN (Samangouei et al. 2018) # unavailable in ART?
Virtual adversarial training (Miyato et al., 2015) # unavailable in ART?
"""
