from itertools import combinations

import numpy as np
import torch
import glob
from torch import Tensor
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt

def get_out_features(sequential: torch.nn.Module, img_shape: tuple):
    # Calculate the output shape of the last convolutional layer
    dummy_input = torch.zeros(1, img_shape[0], img_shape[1], img_shape[2])
    with torch.no_grad():
        dummy_output = sequential(dummy_input)
        
    out_shape = torch.tensor(dummy_output.shape)

    # Calculate the input size for the fully connected layers
    return torch.prod(out_shape)

def get_label(path: str, dataset_name: str):
    dataset_split = path.split(dataset_name,1)[1]
    label = int(dataset_split.split('/',2)[1])
    return label - 1

def getDataset(dataset_dir: str, dataset_name: str):
    file_paths = np.sort(glob.glob(dataset_dir + '/*/*.bmp'))
    labels = np.array([get_label(path, dataset_name) for path in file_paths], np.float32)
    
    return file_paths, labels

def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix

def stratifiedSortedSplit(file_paths: np.array, labels: np.array, 
                    train_size: float, test_size: float, random_state: int):
    """Splits image paths and labels equally for each class, then sorts them"""
    splitter = StratifiedShuffleSplit(n_splits=1, 
                                      train_size=train_size, test_size=test_size, random_state=random_state)
    train_indices, test_indices = next(splitter.split(file_paths, labels))
    
    files_train, labels_train = file_paths[train_indices], labels[train_indices]
    files_test, labels_test = file_paths[test_indices], labels[test_indices]

    sort_index = np.argsort(labels_train)
    labels_train = labels_train[sort_index]
    files_train = files_train[sort_index]

    sort_index = np.argsort(labels_test)
    labels_test = labels_test[sort_index]
    files_test = files_test[sort_index]

    labels_train: Tensor = torch.from_numpy(labels_train)
    labels_test: Tensor = torch.from_numpy(labels_test)
    
    return files_train, labels_train, files_test, labels_test

def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    plt.figure(figsize=(5,5))
    
    classes = np.unique(targets.astype(int))[:10]
    
    legends = [str(i+1) for i in classes]

    for i in classes:
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5)
            
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(legends)
    
    plt.show()

def extract_embeddings(dataloader, model, device):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            images = images.to(device)
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels