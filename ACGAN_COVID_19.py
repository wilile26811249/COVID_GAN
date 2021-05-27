import enum
import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

os.makedirs("weights", exist_ok = True)
os.makedirs("samples", exist_ok = True)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type = int, default = 5000, help = "Number of epochs for training")
parser.add_argument("--batch-size", type = int, default = 32, help = "Size of each batches")
parser.add_argument("--classes", type = int, default = 4, help = "Number of classes for your dataset")
parser.add_argument("--latent-dim", type = int, default = 110, help = "Dimension of the latent vector")
parser.add_argument("--img-size", type = int, default = 128, help = "Size of each img dimension")
parser.add_argument("--img-channel", type = int, default = 3, help = "Number of the channel for the image")
parser.add_argument("--data-dir", type = str, default = "/data", help = "Data root dir of your training data")
parser.add_argument("--sample-interval", type = int, default = 100, help = "Interval for sampling image from generator")
parser.add_argument("--gpu-id", type = int, default = 1, help = "Select the specific gpu to training")
arg = parser.parse_args()
print(f"Training Hyperparameters: {arg}")

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else "cpu")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    """
    Conditional Image Synthesis with Auxiliary Classifier GANs
    Paper: https://arxiv.org/pdf/1610.09585.pdf

    Return
    ======
    out: int, shape(batch_size, 3, 128, 128)
    """
    def __init__(self):
        super().__init__()
        def generator_block(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size = 5,
                    stride = 2, padding = 2, output_padding = 1, bias = False),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2, inplace = True)
            )
        self.label_to_latent = nn.Embedding(arg.classes, arg.latent_dim)
        self.init_img_size = arg.img_size // 8
        self.layer1 = nn.Linear(arg.latent_dim, 768 * self.init_img_size ** 2)
        self.layer2 = nn.Sequential(
            generator_block(768, 384),   # (768,  8,  8) -> (384, 16, 16)
            generator_block(384, 256),   # (384, 16, 16) -> (256, 32, 32)
            generator_block(256, 192),   # (256, 32, 32) -> (191, 64, 64)
            nn.ConvTranspose2d(192, 3, kernel_size = 5,
                stride = 2, padding = 2, output_padding = 1),
            nn.Tanh()
        )
        self.apply(weights_init)


    def forward(self, noise, label):
        # Convert number of classes to latent space
        latent_vector = torch.mul(self.label_to_latent(label), noise)
        out = self.layer1(latent_vector)
        out = out.view(out.shape[0], 768, self.init_img_size, self.init_img_size)
        out = self.layer2(out)
        return out


class Discriminator(nn.Module):
    """
    Conditional Image Synthesis with Auxiliary Classifier GANs
    Paper: https://arxiv.org/pdf/1610.09585.pdf

    Return
    ======
    out: int, shape(batch_size, 3, 128, 128)
    """
    def __init__(self):
        super().__init__()
        def discriminator_block(in_dim, out_dim, stride, bn = True):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 3, stride = stride, padding = 1),
                nn.BatchNorm2d(out_dim) if bn else nn.Identity(),
                nn.LeakyReLU(0.2, inplace = True)
            )
        self.conv_block = nn.Sequential(
            discriminator_block(3, 16, 2, False),
            discriminator_block(16, 32, 1),
            discriminator_block(32, 64, 2),
            discriminator_block(64, 128, 1),
            discriminator_block(128, 256, 2),
            discriminator_block(256, 512, 1)
        )

        self.downsample_size = arg.img_size // 8
        self.adversarial_layer = nn.Sequential(
            nn.Linear(512 * self.downsample_size ** 2, 1),
            nn.Sigmoid()
        )
        self.classifier_layer = nn.Linear(512 * self.downsample_size ** 2, arg.classes)

    def forward(self, img):
        out = self.conv_block(img)
        out = out.view(out.shape[0], -1)
        adversarial_result = self.adversarial_layer(out)
        classifier_result = self.classifier_layer(out)
        return adversarial_result, classifier_result


# Loss Function
adversarial_loss = nn.BCELoss()
auxiliary_loss = nn.CrossEntropyLoss()

# Initialize Generator and Discriminator
netG = Generator()
netD = Discriminator()

# Convert to gpu or not
if cuda:
    netG.cuda()
    netD.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

# Initialize optimizer for generator and discriminator
optim_G = torch.optim.Adam(netG.parameters(), lr = 0.0002, beta = (0.5, 0.999))
optim_D = torch.optim.Adam(netD.parameters(), lr = 0.0002, beta = (0.5, 0.999))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def generate_img(n_row, steps):
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, arg.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = netG(z, labels)
    torch.utils.save_image(gen_imgs.data, "samples/%d.png" % steps, nrow = n_row, normalize = True)


# Configure dataloader
covid_transform = T.Compose([
    T.Resize(arg.img_size),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
covid_dataset = ImageFolder(arg.data_dir, transform = covid_transform)
covid_dataloader = torch.utils.data.DataLoader(
    covid_dataset,
    batch_size = arg.batch_size,
    shuffle = True,
    num_workers = 4 if cuda else 0
)

#---------------
# Trainging
#---------------
for epoch in range(arg.epochs):
    for i, (imgs, labels) in enumerate(covid_dataloader):
        real_imgs = imgs.to(device)
        labels = labels.to(device)

        batch_size = imgs.shape[0]

        # Define adversarial label
        real = torch.ones((batch_size)).to(device)
        fake = torch.zeros((batch_size)).to(device)


        #--------------------
        # Train Generator
        #--------------------
        # Generate noise and label
        z = Variable(torch.randn(batch_size)).to(device)
        gen_labels = Variable(LongTensor(np.random.randint(0, arg.classes, batch_size)))

        # Generate images
        gen_imgs = netG(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        fake_adv, fake_aux = netD(gen_imgs)
        g_loss = 0.5 * (adversarial_loss(fake_adv, real) + auxiliary_loss(fake_aux, gen_labels))

        g_loss.backward()
        optim_G.step()


        #--------------------
        # Train Discriminator
        #--------------------
        optim_D.zero_grad()

        # Loss for real images
        real_adv, real_aux = netD(real_imgs)
        d_real_loss = (adversarial_loss(real_adv, real) + auxiliary_loss(real_aux, labels)) / 2

        # Loss for fake images
        fake_adv, fake_aux = netD(gen_imgs.detach())
        d_fake_loss = (adversarial_loss(fake_adv, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2

        # Total discriminator loss
        d_loss = d_real_loss + d_fake_loss

        # Calculate discriminator accuracy
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis = 0)
        gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis = 0)
        d_acc = np.mean(np.argmax(pred, axis = 1) == gt)

        d_loss.backward()
        optim_D.step()

    print(
        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
        % (epoch, arg.epochs, i, len(covid_dataloader), d_loss.item(), 100 * d_acc, g_loss.item())
    )

    steps = epoch * len(covid_dataloader) + i
    if steps % arg.sample_interval == 0:
        generate_img(n_row = 10, steps = steps)
