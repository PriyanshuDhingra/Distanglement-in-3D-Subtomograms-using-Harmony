import torch
import numpy as np
from model3D.model import get_instance_model_optimizer, load_ckp
from model3D.data import data_loader, estimate_optimal_gamma
from model3D.train import train_model
from model3D.evaluate import evaluate_model
import argparse
import warnings

warnings.filterwarnings('ignore')

def train_and_evaluate(dataset_name, batch_size = 100, n_epochs = 5, learning_rate= 0.0001, z_dim=2, pixel = 64, load_model = False, w=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    siamese, optimizer = get_instance_model_optimizer(device,learning_rate,z_dim,pixel)
    train_loader, test_loader, mu, std = data_loader(dataset_name, pixel, batch_size)
    if load_model:
        siamese, optimizer, start_epoch, epoch_train_loss, epoch_valid_loss, valid_loss_min = load_ckp(siamese, optimizer, 'best_model_Harmony_3D_' + dataset_name + '_z_dim_{}.pt'.format(z_dim))
    else:
        valid_loss_min = np.inf
        start_epoch = 0
        epoch_train_loss = []
        epoch_valid_loss = []
    train_model(dataset_name, siamese, optimizer, train_loader, test_loader, device, start_epoch, n_epochs, epoch_train_loss, epoch_valid_loss, valid_loss_min, z_dim, pixel, batch_size, w)
    evaluate_model(dataset_name, siamese, z_dim, pixel, batch_size, device=device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train and Evaluate Harmony on your dataset')
    parser.add_argument('-z', '--z-dim', type=int, default=1)
    parser.add_argument('-bs', '--batch-size', type=int, default=50)
    parser.add_argument('-ep', '--num-epochs', type=int, default=200)
    parser.add_argument('-l', '--learning-rate', type=float, default=0.0001)
    parser.add_argument('--load-model', action='store_true')
    parser.add_argument('-w','--gamma',type=int)
    parser.add_argument('-dat', '--dataset', type=str)
    parser.add_argument('-p', '--pixel', type=int, required=False)
    args = parser.parse_args()
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    z_dim = args.z_dim
    dataset_name = args.dataset
    pixel = args.pixel
    w = args.gamma
    load_model = False
    if args.load_model:
        load_model = True

    if args.pixel:
        print('The number of voxel per dimension is limited to 32 for this version')

    pixel = 32

    if args.gamma:
        w = args.gamma
    else:
        w = estimate_optimal_gamma(dataset_name, batch_size)

    train_and_evaluate(dataset_name=dataset_name, batch_size=batch_size, n_epochs=num_epochs,
                       learning_rate=learning_rate, z_dim=z_dim, pixel=pixel, load_model=load_model, w=w)