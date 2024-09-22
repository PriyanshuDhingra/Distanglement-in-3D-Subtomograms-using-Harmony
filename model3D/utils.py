import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
import numpy as np
from model3D.model import load_ckp
from scipy.stats import norm
from scipy.special import expit
import time
import kornia
import torch.nn as nn


#
def loss_fn(image_z1, image_z2, image_x_theta1, image_x_theta2, phi1, phi2, dim, w):
    n = image_x_theta1.size(0)
    # recon_loss1 and recon_loss2 are mean squared error (MSE) losses that measure how well the reconstructed images (image_z1 and image_z2) match the transformed images (image_x_theta1 and image_x_theta2).
    recon_loss1 = F.mse_loss(image_z1, image_x_theta1, reduction="sum").div(n)
    recon_loss2 = F.mse_loss(image_z2, image_x_theta2, reduction="sum").div(n)
    # This penalizes any discrepancies between the two transformed images (image_x_theta1 and image_x_theta2), enforcing consistency between them.
    branch_loss = F.mse_loss(image_x_theta1, image_x_theta2, reduction="sum").div(n)

    # This computes the KL divergence between two multivariate normal distributions (one for each image’s latent space, z1 and z2). The mean and variance of the latent variables are extracted from phi1 and phi2, and used to construct these distribution
    z1_mean = phi1[:, 6:6 + dim]
    z1_var = phi1[:, -dim:]
    z2_mean = phi2[:, 6:6 + dim]
    z2_var = phi2[:, -dim:]
    dist_z1 = torch.distributions.multivariate_normal.MultivariateNormal(z1_mean, torch.diag_embed(z1_var.exp()))
    dist_z2 = torch.distributions.multivariate_normal.MultivariateNormal(z2_mean, torch.diag_embed(z2_var.exp()))
    z_loss = torch.mean(torch.distributions.kl.kl_divergence(dist_z1, dist_z2)).div(dim)
    # The term w balances the reconstruction losses and the KL divergence. It's crucial to set this weight properly to avoid overfitting to the reconstruction loss or the latent space regularization.
    loss = w * (recon_loss1 + recon_loss2 + branch_loss) + z_loss
    # The total loss combines the reconstruction losses, branch loss, and KL divergence, weighted by w. This ensures that the model learns both to reconstruct the input images and to properly disentangle transformations from the semantic content in the latent space.
    return loss


def plot_loss(epoch_train_loss, epoch_valid_loss):
    fig, ax = plt.subplots(dpi=150)
    train_loss_list = [x for x in epoch_train_loss]
    valid_loss_list = [x for x in epoch_valid_loss]
    line1, = ax.plot([i for i in range(len(train_loss_list))], train_loss_list)
    line2, = ax.plot([i for i in range(len(valid_loss_list))], valid_loss_list)
    ax.set_xlabel('Number of epochs')
    ax.set_ylabel('Loss')
    ax.legend((line1, line2), ('Train Loss', 'Validation Loss'))
    plt.savefig("Harmony_mnist_loss_curves.png", bbox_inches="tight")


def _save_sample_images(dataset_name, batch_size, recon_image, image, pixel, mu=None, std=None):
    sample_out = recon_image.reshape(batch_size, pixel * pixel * pixel).astype(np.float32)
    sample_out = sample_out.reshape(batch_size, pixel, pixel, pixel)
    plt.clf()
    fig = plt.figure(figsize=(8, 8))  # Notice the equal aspect ratio
    plot_per_row = max(1, int(batch_size / 10))  # Ensure at least 1 row
    plot_per_col = max(1, int(batch_size / 10))  # Ensure at least 1 column
    ax = [fig.add_subplot(plot_per_row, plot_per_col, i + 1) for i in range(batch_size)]

    i = 0
    for a in ax:
        a.xaxis.set_visible(False)
        a.yaxis.set_visible(False)
        a.set_aspect('equal')
        a.imshow(np.mean(sample_out[i], axis=2), cmap='binary')
        i += 1

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("Harmony_decoded_image_sample_" + dataset_name + ".png", bbox_inches="tight")


    sample_in = image.reshape(batch_size, pixel * pixel * pixel).astype(np.float32)
    sample_in = sample_in.reshape(batch_size, pixel, pixel, pixel)
    plt.clf()
    fig = plt.figure(figsize=(8, 8))  # Notice the equal aspect ratio
    plot_per_row = int(batch_size / 10)
    plot_per_col = int(batch_size / 10)
    ax = [fig.add_subplot(plot_per_row, plot_per_col, i + 1) for i in range(batch_size)]

    i = 0
    for a in ax:
        a.xaxis.set_visible(False)
        a.yaxis.set_visible(False)
        a.set_aspect('equal')
        a.imshow(np.mean(sample_in[i], axis=2), cmap='binary')
        i += 1

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("Harmony_input_image_sample" + dataset_name + ".png", bbox_inches="tight")



# This function generates and visualizes images from the latent space learned by the model. It explores the latent manifold by sampling different latent variables (z) and decoding them back into images.
# For 1D or 2D latent spaces, it samples a grid of points (z_arr) from the standard normal distribution and passes them through the decoder to generate images.
# The generated images represent interpolations across the latent space, which can reveal how smoothly the model encodes and decodes variations in the subtomograms.
def generate_manifold_images(dataset_name, trained_vae, pixel, z_dim=1, batch_size=100, device='cuda'):
    trained_vae.eval()
    decoder = trained_vae.autoencoder.decoder
    if z_dim > 2:
        print("Generation of manifold image for higher than 2-dimension is not implemented in this version")
        print("Manifold images not saved")
        return
    elif z_dim == 1:
        z_arr = norm.ppf(np.linspace(0.05, 0.95, batch_size))
    else:
        n = int(np.sqrt(batch_size))
        grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
        z_list = []
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]])
                z_list.append(z_sample)
        z_arr = np.array(z_list).squeeze()

    z = torch.from_numpy(z_arr).float().to(device=device)
    if z_dim == 1:
        z = torch.unsqueeze(z, 1)
    print(z.shape)
    image_z = decoder(z)
    manifold = image_z.cpu().detach().numpy()
    sample_out = manifold.reshape(batch_size, pixel, pixel, pixel).astype(np.float32)
    plt.clf()
    fig = plt.figure(figsize=(8, 8))  # Notice the equal aspect ratio
    plot_per_row = int(batch_size / 10)
    plot_per_col = int(batch_size / 10)
    ax = [fig.add_subplot(plot_per_row, plot_per_col, i + 1) for i in range(batch_size)]

    i = 0
    for a in ax:
        a.xaxis.set_visible(False)
        a.yaxis.set_visible(False)
        a.set_aspect('equal')
        a.imshow(np.mean(sample_out[i], axis=2), cmap='binary')
        i += 1

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("Harmony_manifold_image_" + dataset_name + ".png", bbox_inches="tight")


def plot_sample_images(dataset_name, test_loader, trained_model, pixel, batch_size=100, device='cuda', mu=None,
                       std=None):
    trained_model.eval()
    for batch_idx, images in enumerate(test_loader):
        with torch.no_grad():
            images = images.to(device=device)
            data = images.reshape(batch_size, 1, pixel, pixel, pixel)
            image_z1, image_z2, image_x_theta1, image_x_theta2, phi1, phi2 = trained_model(data)
            break
    pose_image = image_z1.cpu().detach().numpy()
    input_image = data.cpu().detach().numpy()
    with open(dataset_name + 'pose_image.pkl', 'wb') as f:
        pickle.dump(pose_image, f)
    _save_sample_images(dataset_name, batch_size, pose_image, input_image, pixel, mu[:batch_size], std[:batch_size])


def save_output_images(dataset_name, test_loader, trained_model, pixel, type='test', batch_size=100, device='cuda'):
    trained_model.eval()
    all_images = []
    for batch_idx, images in enumerate(test_loader):
        with torch.no_grad():
            images = images.to(device=device)
            data = images.reshape(batch_size, 1, pixel, pixel, pixel)
            image_z1, image_z2, image_x_theta1, image_x_theta2, phi1, phi2 = trained_model(data)
            pose_image = image_z1.cpu().detach().numpy()
            all_images.append(pose_image)
    output_image = np.array(all_images)
    f = open('Harmony_decoded_images_' + dataset_name + '_' + type + '.pkl', 'wb')
    pickle.dump(output_image, f)
    f.close()

# This function saves the latent representations (phi1, phi2) of the subtomograms. These latent variables are the core representations that encode both the semantic content and transformation parameters.
def save_latent_variables(dataset_name, data_loader, siamese, type, pixel, z_dim, batch_size=100, device='cuda'):
    Allphi = []
    siamese.eval()
    count = 0
    for batch_idx, images in enumerate(data_loader):
        count += 1
        with torch.no_grad():
            images = images.to(device=device)
            data = images.reshape(batch_size, 1, pixel, pixel, pixel)
            image_z1, image_z2, image_x_theta1, image_x_theta2, phi1, phi2 = siamese(data)
            phi_np = phi1.cpu().detach().numpy()
            Allphi.append(phi_np)
    PhiArr = np.array(Allphi).reshape(count * batch_size, -1)
    filepath = 'Harmony_latent_factors' + dataset_name + '_' + type + 'z_dim_' + str(z_dim) + '.np'
    np.savetxt(filepath, PhiArr)