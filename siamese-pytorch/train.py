import torch
import numpy as np
import yaml
import os
import randomname
import random
from datasets import ImageDataset, TripletDataset
import utils
from torch.utils.data import DataLoader
from networks import EmbeddingNet, TripletNet
from losses import TripletLoss
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from metrics import AccumulatedAccuracyMetric, AverageNonzeroTripletsMetric, TripletAccuracy

def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, device, final_path, n_epochs, log_interval, 
        save_best, save_after, accuracy_margin = 2, metrics=[], start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    
    metrics_funcs = []
    
    for metric in metrics:
        if metric == 'accuracy':
            metrics_funcs.append(AccumulatedAccuracyMetric())
        elif metric == 'nonzero-triplets':
            metrics_funcs.append(AverageNonzeroTripletsMetric())
        elif metric == 'triplet-accuracy':
            metrics_funcs.append(TripletAccuracy(accuracy_margin))
    
    # Initialize TensorBoard
    writer = SummaryWriter(final_path)

    best_val = 10000000000
    for epoch in range(start_epoch, n_epochs):

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, 
                                          optimizer, device, log_interval, metrics_funcs)
        
        scheduler.step()
        
        writer.add_scalar('train_loss', train_loss, epoch)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {:.2f}'.format(metric.name(), metric.value())
            writer.add_scalar(f'train_{metric.name().lower()}', metric.value(), epoch)

        val_loss, metrics = test_epoch(val_loader, model, loss_fn, device, metrics_funcs)
        val_loss /= len(val_loader) if len(val_loader) != 0 else 1
        
        writer.add_scalar('val_loss', val_loss, epoch)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
            writer.add_scalar(f'val_{metric.name().lower()}', metric.value(), epoch)

        print(message)
        
        # Update "best.pt" model if val_loss in current epoch is lower than the best validation loss
        if save_best and val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(final_path, "best.pt")
            )

        # Save model based on the frequency defined by "args['save_after']"
        if save_after != 0 and (epoch + 1) % save_after == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(final_path, "epoch_{}.pt".format(epoch + 1))
            )

def train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval, metrics_funcs):
    for metric in metrics_funcs:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0
    batch_idx = 0

    for idx, (data, target) in enumerate(train_loader):
        batch_idx = idx
        target = target.type(torch.LongTensor).to(device) if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)

        data = tuple(d.to(device) for d in data)

        optimizer.zero_grad()
        outputs = model(*data)
        
        if type(outputs) not in (tuple, list):
            outputs = (outputs,)
        
        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics_funcs:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics_funcs:
                message += '\t{}: {:.2f}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics_funcs

def test_epoch(val_loader, model, loss_fn, device, metrics_funcs):
    with torch.no_grad():
        for metric in metrics_funcs:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target.type(torch.LongTensor).to(device) if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)

            data = tuple(d.to(device) for d in data)

            outputs = model(*data)
            
            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics_funcs:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics_funcs

if __name__ == "__main__":
    torch.cuda.empty_cache()
    
    # Open config
    with open('config.yaml', 'r') as file:
        args = yaml.safe_load(file)
        
        # Assign a unique folder as training output
    training_id = f'{randomname.get_name()}-{str(random.randint(1,9))}'
    print('-----------------------------')
    print(f'Beginning training with id {training_id}!')

    final_path = os.path.join(
        args['output_path'], training_id)

    while os.path.isdir(final_path):
        training_id = f'{randomname.get_name()}-{str(random.randint(1,9))}'
        final_path = os.path.join(
            args['output_path'], training_id)

    os.makedirs(final_path, exist_ok=True)

    # Write config to output folder
    with open(os.path.join(final_path, 'config.yaml'), 'w') as file:
        yaml.dump(args, file)

    # Set device to CUDA if a CUDA device is available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get image paths and labels from dataset
    dataset_path = os.path.join(args['dataset_dir'], args['dataset_name'])
    file_paths, labels = utils.getDataset(dataset_path, args['dataset_name'])

    # Split image paths and labels using Stratified
    files_train, labels_train, files_test, labels_test = utils.stratifiedSortedSplit(
        file_paths, labels, args['train_size'], args['test_size'], args['random_seed'])

    train_count = np.unique(labels_train, return_counts=True)[1].mean()
    test_count = np.unique(labels_test, return_counts=True)[1].mean()
    print(
        f'Split {train_count} images from each class for train and {test_count} for test')
    
    train_dataset = ImageDataset(
        files_train, labels_train, args['final_shape'], **args['train_dataset'])
    test_dataset = ImageDataset(
        files_test, labels_test, args['final_shape'], **args['test_dataset'])
    
    train_loader = DataLoader(train_dataset, **args['train_dataloader'])
    test_loader = DataLoader(test_dataset, **args['test_dataloader'])
    
    triplet_train_dataset = TripletDataset(
        files_train, labels_train, args['final_shape'], args['random_seed'], **args['train_dataset']) # Returns triplets of images
    triplet_test_dataset = TripletDataset(
        files_test, labels_test, args['final_shape'], args['random_seed'], **args['test_dataset'])
    triplet_train_loader = DataLoader(triplet_train_dataset, **args['train_dataloader'])
    triplet_test_loader = DataLoader(triplet_test_dataset, **args['test_dataloader'])
    
    embedding_net = EmbeddingNet(args['final_shape'], args['output_num'])
    model = TripletNet(embedding_net)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])
    scheduler = lr_scheduler.StepLR(optimizer, **args['lr_scheduler'])
    loss_fn = TripletLoss(args['loss_margin'])
    
    fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, device, final_path, **args['fit'])
    
    train_embeddings_baseline, train_labels_baseline = utils.extract_embeddings(train_loader, model, args['output_num'], device)
    utils.plot_embeddings(train_embeddings_baseline, train_labels_baseline)
    val_embeddings_baseline, val_labels_baseline = utils.extract_embeddings(test_loader, model, args['output_num'], device)
    utils.plot_embeddings(val_embeddings_baseline, val_labels_baseline)