import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from loaddata import load_cifar, load_mnist
#from art.attacks import FastGradientMethod # ART ver 1.1.0
from art.attacks.evasion import FastGradientMethod # ART 1.3.0
from tests.utils import get_image_classifier_pt

# load pytorch mnist model available in ART
classifier = get_image_classifier_pt()

class DefenseVAE(nn.Module):
    def __init__(self, imageShape):
        super(DefenseVAE, self).__init__()
 
        # Encoder
        self.conv0 = nn.Conv2d(imageShape[0], 64, kernel_size=5, stride=1, padding=2, bias= False)
        self.conv0_bn = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=3, bias= False)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias= False)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias= False)
        self.conv3_bn = nn.BatchNorm2d(256)
        # Latent space
        self.fc21 = nn.Linear(4096, 128)
        self.fc22 = nn.Linear(4096, 128)
 
        # Decoder
        self.fc3 = nn.Linear(128, 4096)
        self.fc3_bn = nn.BatchNorm1d(4096)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias= False)
        self.deconv1_bn = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias= False)
        self.deconv2_bn = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=3, bias= False)
        self.deconv3_bn = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 1, kernel_size=5, stride=1, padding=2, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
 
    def encode(self, x):
        out = self.relu(self.conv0_bn(self.conv0(x)))
        out = self.relu(self.conv1_bn(self.conv1(out)))
        out = self.relu(self.conv2_bn(self.conv2(out)))
        out = self.relu(self.conv3_bn(self.conv3(out)))
        h1 = out.view(out.size(0), -1)
        return self.fc21(h1), self.fc22(h1)
 
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
 
    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        out = h3.view(h3.size(0), 256, 4, 4)
        out = self.relu(self.deconv1_bn(self.deconv1(out)))
        out = self.relu(self.deconv2_bn(self.deconv2(out)))
        out = self.relu(self.deconv3_bn(self.deconv3(out)))
        out = self.sigmoid(self.deconv4(out))
        return out
 
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        rec = self.decode(z)
        return rec, mu, logvar


def loss_lambda(recon_x, x, mu, logvar, r):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + r * KLD, BCE, KLD

def loss_lambda_e2e(recon_x, x, mu, logvar, r, classifier, y):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    y_pred = classifier.predict(recon_x.detach().numpy())
    y_pred = torch.FloatTensor(y_pred)
    y_pred = torch.autograd.Variable(y_pred)
    model_loss = nn.CrossEntropyLoss(reduction='sum')(y_pred, y)

    return BCE + r * KLD + model_loss, BCE, KLD, model_loss


def train(epoch, classifier):
    model.train()
    train_loss = 0

    loss_ll = []
    loss_B = []
    loss_K = []
    for batch_idx, (adv_batch, clean_batch, y_batch) in enumerate(train_data_loader):
        adv_batch = adv_batch.to(device)
        clean_batch = clean_batch.to(device)
        y_batch = y_batch.to(device)
        y_batch = torch.argmax(y_batch, dim=1)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(adv_batch)

        #loss, BCE, KLD = loss_lambda(recon_batch, clean_batch, mu, logvar, 0.5) # loss for simple Defense VAE
        loss, BCE, KLD, model_loss = loss_lambda_e2e(recon_batch, clean_batch, mu, logvar, 0.5, classifier, y_batch)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(adv_batch), len(train_data_loader.dataset),
                       100. * batch_idx / len(train_data_loader), loss.item() / len(adv_batch)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_data_loader.dataset)))

batch_size = 128
nb_epochs = 10
# load data
x_train, y_train, x_test, y_test = load_mnist()
x_train = x_train.transpose(0, 3, 1, 2)
x_train = x_train[0:10000] # temporary
y_train = y_train[0:10000] # temporary
# generate adversarial images
attack = FastGradientMethod(classifier, eps=0.1)
x_train_adv = attack.generate(x_train)

# generate train loader
x_train = torch.tensor(x_train).type(torch.FloatTensor)
x_train_adv = torch.tensor(x_train_adv).type(torch.FloatTensor)
y_train = torch.tensor(y_train).type(torch.FloatTensor)

train_data = torch.utils.data.TensorDataset(x_train_adv, x_train, y_train)
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

device = torch.device("cpu")
model = DefenseVAE(x_train.shape[1:]).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

if __name__ == "__main__":
    for epoch in range(1, nb_epochs + 1):
        train(epoch, classifier)
