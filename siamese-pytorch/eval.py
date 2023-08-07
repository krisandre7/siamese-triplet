import torch
import yaml
import os
import utils
import numpy as np
from datasets import ImageDataset, EmbeddingsDataset
from torch.utils.data import DataLoader
import argparse
from networks import EmbeddingNet, TripletNet, ClassificationPerceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from train import fit
from losses import TripletLoss
from torch import optim
from torch.optim import lr_scheduler 
from metrics import AccumulatedAccuracyMetric
import shutil

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
        train_config = yaml.safe_load(file)

    eval_id = args.checkpoint_dir.split('/')[-1]
    final_path = os.path.join('eval/', eval_id)
    
    try:
        os.makedirs(final_path)
    except:
        shutil.rmtree(final_path)
        os.makedirs(final_path)
        

    # Set device to CUDA if a CUDA device is available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get image paths and labels from dataset
    dataset_path = os.path.join(train_config['dataset_dir'], train_config['dataset_name'])
    file_paths, labels = utils.getDataset(dataset_path, train_config['dataset_name'])

    # Split image paths and labels using Stratified
    files_train, labels_train, files_test, labels_test = utils.stratifiedSortedSplit(
        file_paths, labels, train_config['train_size'], train_config['test_size'], train_config['random_seed'])

    train_count = np.unique(labels_train, return_counts=True)[1].mean()
    test_count = np.unique(labels_test, return_counts=True)[1].mean()
    print(
        f'Split {train_count} images from each class for train and {test_count} for test')
    
    train_dataset = ImageDataset(
        files_train, labels_train, train_config['final_shape'], **train_config['train_dataset'])
    test_dataset = ImageDataset(
        files_test, labels_test, train_config['final_shape'], **train_config['test_dataset'])
    
    train_loader = DataLoader(train_dataset, **train_config['train_dataloader'])
    test_loader = DataLoader(test_dataset, **train_config['test_dataloader'])
    
    embedding_net = EmbeddingNet(train_config['final_shape'], train_config['output_num'])
    
    trained_model = TripletNet(embedding_net)
    trained_model.to(device)
    
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, args.checkpoint_name))
    trained_model.load_state_dict(checkpoint['model_state_dict'])
    
    # n_classes = len(np.unique(labels))
    # model = ClassificationPerceptron(train_config['output_num'], n_classes)
    # model.to(device)
    
    X_train, y_train = utils.extract_embeddings(train_loader, trained_model, train_config['output_num'], device)
    # embed_train = EmbeddingsDataset(X_train, y_train)
    # train_loader = DataLoader(embed_train, **train_config['train_dataloader'])    
    # figure = utils.plot_embeddings(X_train, y_train)
    # figure.savefig(os.path.join(args.out_path, 'train.png'))
    
    X_test, y_test = utils.extract_embeddings(test_loader, trained_model, train_config['output_num'], device)
    # embed_test = EmbeddingsDataset(X_test, y_test)
    # test_loader = DataLoader(embed_test, **train_config['test_dataloader'])
    # figure = utils.plot_embeddings(X_test, y_test)
    # figure.savefig(os.path.join(args.out_path, 'test.png'))
    
    
    # optimizer = optim.Adam(model.parameters(), lr=train_config['learning_rate'])
    # scheduler = lr_scheduler.StepLR(optimizer, **train_config['lr_scheduler'])
    # loss_fn = torch.nn.CrossEntropyLoss()
    
    # train_config['fit']['metrics'] = ['accuracy']
    
    # fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, device, final_path, **train_config['fit'])
    
    X = np.concatenate((X_train, X_test))
    X_tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(X)
    figure = plt.figure(figsize=(5,5))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.5)
    plt.show()
    
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test) 

    print(f'Accuracy: {accuracy_score(y_test, y_pred)}%')
    # print(f"Precision {precision_score(y_test, y_pred, average='macro')}")
    # print(f"Recall: {recall_score(y_test, y_pred, average='macro')}")
    # print(f"F1: {f1_score(y_test, y_pred, average='macro')}")
    # conf_matrix = confusion_matrix(y_test, y_pred, normalize='all')
    # disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    
    # disp.plot()
    # plt.savefig(os.path.join(args.out_path, 'confusion_matrix.png'))

    svm = make_pipeline(StandardScaler(), SVC(gamma='auto', random_state=train_config['random_seed']))
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf', 'poly']} 
  
    grid = GridSearchCV(SVC(), param_grid, refit = True, cv=2)
    
    # fitting the model for grid search
    grid.fit(X_train, y_train)
    svm = make_pipeline(StandardScaler(), SVC(random_state=train_config['random_seed'], **grid.best_params_))
    svm.fit(X_train, y_train)
    
    y_pred = svm.predict(X_test)
    print(accuracy_score(y_test, y_pred))