import pickle
import torch
import numpy as np
import math

def loadpickle(fname):
    with open(fname, 'rb') as F:
        array = pickle.load(F)
    F.close
    return array

# Estimating the optimal value of the hyperparameter gamma which controls the balance between the reconstruction loss and KL divergence.
# The estimation depends on the size of the training dataset (M) and the batch size. By dividing the dataset size by a factor, it approximates a suitable gamma value based on how many batches the dataset is split into.
def estimate_optimal_gamma(dataset_name, batch_size):
    X_train = loadpickle('data/'+ dataset_name+ '_train.pkl')
    M = X_train.shape[0]
    N = batch_size * 1000
    return math.ceil(M / N)

def data_loader(dataset_name,pixel,batch_size = 100, shuffle = True, normalize = True):
    X_train = loadpickle('data/' + dataset_name + '_train.pkl')
    X_test = loadpickle('data/' + dataset_name + '_test.pkl')
    if normalize:
        print('Normalizng Particles')
        # The datasets are normalized, which is critical for stable training of the model. Normalization ensures that the intensity values of 3D subtomograms are standardized, making it easier for the model to learn consistent representations.
        mu = X_train.reshape(-1, pixel * pixel * pixel).mean(1)
        std = X_train.reshape(-1, pixel * pixel * pixel).std(1)
        X_train = (X_train - mu[:, np.newaxis, np.newaxis, np.newaxis]) / std[:, np.newaxis, np.newaxis, np.newaxis]
        mu = X_test.reshape(-1, pixel * pixel * pixel).mean(1)
        std = X_test.reshape(-1, pixel * pixel * pixel).std(1)
        X_test = (X_test - mu[:, np.newaxis, np.newaxis, np.newaxis]) / std[:, np.newaxis, np.newaxis, np.newaxis]

    # The data is reshaped to a 5D tensor: (batch_size, channels, depth, height, width), where depth, height, and width represent the 3D dimensions of the subtomograms.
    X_train = X_train.reshape(X_train.shape[0], 1, pixel, pixel, pixel)
    X_test = X_test.reshape(X_test.shape[0], 1, pixel, pixel, pixel)
    train_x = torch.from_numpy(X_train).float()
    test_x = torch.from_numpy(X_test).float()

    # Iteratable Dataset
    train_loader = torch.utils.data.DataLoader(train_x, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_x, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, test_loader, mu, std