import torch
import yaml
import os
import utils
import numpy as np
from datasets import ImageDataset
from torch.utils.data import DataLoader
import argparse
from networks import EmbeddingNet, TripletNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":
    torch.cuda.empty_cache()
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-v',
        '--dataset_dir',
        type=str,
        help="Path to directory containing dataset.",
        default='casia'
    )
    parser.add_argument(
        '-o',
        '--out_path',
        type=str,
        help="Path for saving prediction images.",
        default='./eval'
    )
    parser.add_argument(
        '-c',
        '--checkpoint_dir',
        type=str,
        help="Path of model checkpoint to be used for inference.",
        required=True
    )
    parser.add_argument(
        '-n',
        '--checkpoint_name',
        type=str,
        help="Path of model checkpoint to be used for inference.",
        default='best.pt'
    )    

    args = parser.parse_args()

    # Open config
    with open(os.path.join(args.checkpoint_dir, 'config.yaml'), 'r') as file:
        config = yaml.safe_load(file)

    os.makedirs(args.out_path, exist_ok=True)
    
    final_path = args.checkpoint_dir

    # Set device to CUDA if a CUDA device is available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get image paths and labels from dataset
    dataset_path = os.path.join(config['dataset_dir'], config['dataset_name'])
    file_paths, labels = utils.getDataset(dataset_path, config['dataset_name'])

    # Split image paths and labels using Stratified
    files_train, labels_train, files_test, labels_test = utils.stratifiedSortedSplit(
        file_paths, labels, config['train_size'], config['test_size'], config['random_seed'])

    train_count = np.unique(labels_train, return_counts=True)[1].mean()
    test_count = np.unique(labels_test, return_counts=True)[1].mean()
    print(
        f'Split {train_count} images from each class for train and {test_count} for test')
    
    train_dataset = ImageDataset(
        files_train, labels_train, config['final_shape'], **config['train_dataset'])
    test_dataset = ImageDataset(
        files_test, labels_test, config['final_shape'], **config['test_dataset'])
    
    train_loader = DataLoader(train_dataset, **config['train_dataloader'])
    test_loader = DataLoader(test_dataset, **config['test_dataloader'])
    
    
    embedding_net = EmbeddingNet(config['final_shape'], config['output_num'])
    model = TripletNet(embedding_net)
    model.to(device)
    
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, args.checkpoint_name))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    X_train, y_train = utils.extract_embeddings(train_loader, model, config['output_num'], device)
    # figure = utils.plot_embeddings(X_train, y_train)
    # figure.savefig(os.path.join(args.out_path, 'train.png'))
    X_test, y_test = utils.extract_embeddings(test_loader, model, config['output_num'], device)
    # figure = utils.plot_embeddings(X_test, y_test)
    # figure.savefig(os.path.join(args.out_path, 'test.png'))
    
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test) 

    print(accuracy_score(y_test, y_pred))
    
    svm = make_pipeline(StandardScaler(), SVC(gamma='auto', random_state=config['random_seed']))
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf', 'poly']} 
  
    grid = GridSearchCV(SVC(), param_grid, refit = True, cv=2)
    
    # fitting the model for grid search
    grid.fit(X_train, y_train)
    
    svm = grid.best_estimator_
    
    y_pred = svm.predict(X_test)
    print(accuracy_score(y_test, y_pred))