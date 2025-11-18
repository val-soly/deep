import os 
import tqdm
import shutil
import argparse

import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from models import CNN, ResNet18, FC
from utils import save_model, plot_results, train_iter, test_model, get_optimizer, visualize_results



def main(args):
    print('Training...')

    args.path = os.path.join('experiments', args.path)
    if os.path.exists(args.path):
        print(f'Deleating the existing folder {args.path}')
        shutil.rmtree(args.path)
    os.makedirs(args.path, exist_ok=True)
    print(f'Experiments will be saved in {args.path}')


    # Load the data CIFAR10
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Get a batch of data to visualize the results
    x_test, y_test = next(iter(test_loader))

    # Load the model
    if args.model == 'cnn':
        model = CNN()
    elif args.model == 'resnet18':
        model = ResNet18()
    elif args.model == 'fc':
        model = FC()

   # Set the device (cuda, cpu or mps)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print("Device used: {}".format(device))

    model = model.to(device)

    # Load the optimizer
    optimizer = get_optimizer(model, args.lr, args.optimizer)

    # Train the model
    losses_train = []
    losses_test = []
    acc_train = []
    acc_test = []
    loss, acc = test_model(model, test_loader, device)
    losses_test.append(loss)
    acc_test.append(acc)
    min_loss= 10**10

    # Initialize a progress bar for tracking the progress of epochs
    pbar = tqdm.trange(args.epochs)

    # Loop through the specified number of epochs
    for epoch in pbar:
        # Create a progress bar for the batches in the training loop, set leave=False to avoid cluttering output
        pbar_batch = tqdm.tqdm(train_loader, leave=False)

        # Iterate through the batches in the training data loader
        for i, (x, y) in enumerate(pbar_batch):
            # Perform a single training iteration
            # `train_iter` returns the loss and accuracy for the current batch
            loss, acc = train_iter(model, optimizer, x, y, device)

            # Append the training loss and accuracy to their respective lists for tracking
            losses_train.append(loss)
            acc_train.append(acc)

            # Update the batch progress bar with the current loss and accuracy
            pbar_batch.set_description(f'Loss: {loss:.4f}, Acc: {acc:.2f}')

        # After completing the training for all batches, evaluate the model on the test dataset
        loss, acc = test_model(model, test_loader, device)

        # Append the test loss and accuracy to their respective lists for tracking
        losses_test.append(loss)
        acc_test.append(acc)

        if loss < min_loss:
            best_model = model
            min_loss = loss

        # Update the epoch progress bar with the test loss and accuracy
        pbar.set_description(f'Loss test: {loss:.4f}, Acc test: {acc:.2f}, Best Acc test: {acc_test[np.argmin(losses_test)]:.2f}')

        # Plot the results
        plot_results(losses_train, losses_test, acc_train, acc_test, args)

        # To do 6
        save_model(best_model, args)
        visualize_results(x_test, y_test, best_model, device, args)

    # Save the final results 
    with open(os.path.join(args.path, 'results.txt'), 'a') as f:
        f.write(f'Clean:\n')
        f.write(f'\t Loss: {np.min(losses_test):.4f}, Accuracy: {acc_test[np.argmin(losses_test)]:.2f}\n')

    print('Training done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Train Deep Learning models on CIFAR10""")


    # Training parameters
    parser.add_argument('--path', type=str, default='base', help='Path to save the model and results')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'], help='Optimizer')
    parser.add_argument('--model', type=str, default='fc', help='Model to attack', choices=['cnn', 'resnet18', 'fc'])

    args = parser.parse_args()

    main(args)
