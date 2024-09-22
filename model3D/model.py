import numpy as np
import torch
import kornia # #This library provides efficient operations for geometric transformations (e.g., rotations, translations) on 3D tensors.
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

# Encoder Part of AutoEncoder for processing 3D images by using Conv3D layers
class ConvEncoder(nn.Module):
    def __init__(self, latent_dims, pixel):
        self.pixel = pixel
        super(ConvEncoder, self).__init__()
        # Each convolution layer reduces the spatial dimensions of the input while increasing the number of feature maps (channels).
        # The 3D convolutions process the depth, height, and width of the subtomograms (3D input images), extracting high-level features across all three dimensions.

        #  Takes in 1-channel 3D input (a subtomogram grayscale) and outputs 32 feature maps.
        self.conv1 = nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1)

        #Further reduces the spatial dimensions, outputting 64 feature maps.
        self.conv2 = nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1)  # 7 x 7

        # Continues reducing dimensions, outputting 128 feature maps.
        self.conv3 = nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)  # 4 x 4

        # Brings it down to 512 feature maps with a spatial resolution of 1x1x1 after multiple downsamplings.
        self.conv4 = nn.Conv3d(128, 512, kernel_size=4, stride=1, padding=0)  # 1 x 1

        # Outputs a vector of size (2 * latent_dims + 6).
        self.conv5 = nn.Conv3d(512, 2 * latent_dims + 6, kernel_size=1, stride=1, padding=0)

        # Latent semantic space (2 * latent_dims) which includes both the mean and variance of the latent distribution for reparameterization (required for the Variational Autoencoder (VAE) part).
        # Transformation parameters (6), which represent 3 angles for rotation and 3 translations for 3D transformation disentanglement.

    # Forward Pass
    # Detailed architecture is in supplementary paper
    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.conv3(x))
        x = F.tanh(self.conv4(x))
        x = self.conv5(x)
        phi = x.view(x.size(0), -1)
        # phi is latent representation, consisting of: Transformation parameters (rotation and translation) and Latent variables (mean and variance) for the VAE's reparameterization trick.
        return phi

# Decoder part of the autoencoder, which reconstructs the 3D image from the latent representation. The goal is to reconstruct the original subtomogram using the semantic latent variables.
class ConvDecoder(nn.Module):
    def __init__(self, latent_dims, pixel):
        super(ConvDecoder, self).__init__()
        self.activation = nn.Sigmoid()
        self.pixel = pixel
        self.lt = latent_dims
        # The ConvTranspose3d layers are deconvolution layers or up-sampling layers. They reverse the down-sampling done by the encoder, gradually increasing the spatial dimensions from 1x1x1 back to the original size of the input image.
        # conv: Starts with the latent representation (latent_dims) and expands it into 512 feature maps.
        self.conv = nn.ConvTranspose3d(latent_dims, 512, kernel_size=1, stride=1, padding=0)

        # conv1, conv2, conv3: Gradually upsample the image, eventually returning to the original resolution (pixel x pixel x pixel).
        self.conv1 = nn.ConvTranspose3d(512, 128, kernel_size=4, stride=1, padding=0)
        self.conv2 = nn.ConvTranspose3d(128, 64, kernel_size=4, padding=1, stride=2)
        self.conv3 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1)

        # conv4: Outputs a single-channel 3D volume (since the input subtomograms are grayscale).
        self.conv4 = nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1)

    # Forward Pass: The latent vector z is reshaped into a 3D format (the seed from which the image will be reconstructed). It passes through the deconvolutional layers, progressively upsampling to reconstruct the original subtomogram.
    # The final activation in this case is identity (i.e., no activation) since the reconstructed image is a raw pixel value output.
    def forward(self, z):
        x = z.view(z.size(0), self.lt, 1, 1, 1)
        x = F.tanh(self.conv(x))
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.conv3(x))
        x = self.conv4(x)
        x = x.view(x.size(0), 1, self.pixel, self.pixel, self.pixel)
        return x

# This class is responsible for applying 3D transformations to the subtomograms.
# By applying these transformations, the model learns to reconstruct the subtomogram without being affected by these variations.
class Transformer(object):
    def __call__(self, image, theta, translations):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        B, C, D, H, W = image.size()
        center = torch.tensor([D / 2, H / 2, W / 2]).repeat(B, 1).to(device=device)
        scale = torch.ones(B, 1).to(device=device)
        angle = torch.rad2deg(theta)
        no_trans = torch.zeros(B, 3).to(device=device)
        no_rot = torch.zeros(B, 3).to(device=device)
        # The rotation angles (theta) are used to create an affine transformation matrix using kornia.get_affine_matrix3d.
        M = kornia.geometry.get_affine_matrix3d(translations=no_trans, center=center, scale=scale, angles=angle)
        affine_matrix = M[:, :3, :]
        # The image is then rotated using warp_affine3d, which applies the affine transformation.
        rotated_image = kornia.geometry.warp_affine3d(image, affine_matrix, dsize=(D, H, W), align_corners=False,
                                             padding_mode='zeros')
        N = kornia.geometry.get_affine_matrix3d(translations=translations, center=center, scale=scale, angles=no_rot)
        # For translation
        affine_matrix_tran = N[:, :3, :]
        transformed_image = kornia.geometry.warp_affine3d(rotated_image, affine_matrix_tran, dsize=(D, H, W),
                                                 align_corners=False, padding_mode='zeros')
        return transformed_image


# This is the Variational Autoencoder (VAE) architecture, where the reparameterization trick is used to generate a latent variable z from a mean (mu) and log-variance (logvar). This is part of the VAE’s probabilistic nature.
class AutoEncoder(nn.Module):
    def __init__(self, latent_dims, pixel):
        super(AutoEncoder, self).__init__()
        self.encoder = ConvEncoder(latent_dims, pixel)
        self.decoder = ConvDecoder(latent_dims, pixel)
        self.transform = Transformer()
        self.latent_dims = latent_dims

    # The reparameterization trick ensures that backpropagation is possible by introducing randomness in the latent space sampling, which is key to VAEs.
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    # In the forward pass, the encoder produces the latent variables and the transformation parameters. The image is then decoded from the latent representation, and transformations are applied using the Transformer class.
    def forward(self, x):
        phi = self.encoder(x)
        theta = phi[:, :3]
        trans = phi[:, 3:6]
        z_mu = phi[:, 6:6 + self.latent_dims]
        z_var = phi[:, -self.latent_dims:]
        z = self.reparametrize(z_mu, z_var)
        image_z = self.decoder.forward(z)
        image_x_theta = self.transform(x, theta, trans)
        return image_z, image_x_theta, phi

# This is the Siamese architecture, a crucial part of the cross-contrastive learning. It works by processing both the original image and a transformed version of the image, and enforcing similarity between their latent representations.
class Siamese(nn.Module):
    def __init__(self, latent_dims, pixel):
        super(Siamese, self).__init__()
        self.autoencoder = AutoEncoder(latent_dims, pixel)
        self.transform = Transformer()

    def forward(self, image_z):
        with torch.no_grad():
            angles = torch.FloatTensor(image_z.size(0), 3).uniform_(-np.pi / 2, np.pi / 2).to(device=image_z.device)
            translations = torch.FloatTensor(image_z.size(0), 3).uniform_(-4, 4).to(device=image_z.device)
            transformed_image = self.transform(image_z, angles, translations)
        image_z1, image_x_theta1, phi1 = self.autoencoder(image_z)
        image_z2, image_x_theta2, phi2 = self.autoencoder(transformed_image)
        return image_z1, image_z2, image_x_theta1, image_x_theta2, phi1, phi2


# The forward pass processes two images: Original image (image_z) and Transformed image (created by applying rotation and translation via Transformer).
# It returns the latent representations (phi1, phi2), the decoded images (image_z1, image_z2), and the transformed versions (image_x_theta1, image_x_theta2).
def load_ckp(model, optimizer=None, f_path='./best_model.pt'):
    checkpoint = torch.load(f_path)

    model.load_state_dict(checkpoint['state_dict'])

    optimizer.load_state_dict(checkpoint['optimizer'])

    valid_loss_min = checkpoint['valid_loss_min']
    epoch_train_loss = checkpoint['epoch_train_loss']
    epoch_valid_loss = checkpoint['epoch_valid_loss']

    return model, optimizer, checkpoint['epoch'], epoch_train_loss, epoch_valid_loss, valid_loss_min


def save_ckp(state, f_path='./best_model.pt'):
    torch.save(state, f_path)

# This function returns an instance of the Siamese autoencoder and its optimizer, ready for training.
def get_instance_model_optimizer(device, learning_rate=0.0001, z_dims=2, pixel=64):
    model = Siamese(latent_dims=z_dims, pixel=pixel).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer