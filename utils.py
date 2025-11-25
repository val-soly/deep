import os
import tqdm
import torch 
import numpy as np
from torch import nn
import matplotlib.pyplot as plt


cifar10_classes = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

def save_model(model, args):
    path_model = os.path.join(args.path, 'model.pth')
    torch.save(model.state_dict(), path_model)

def load_model(model, path):
    path_model = os.path.join(path, 'model.pth')
    model.load_state_dict(torch.load(path_model, weights_only=True))
    return model


def get_optimizer(model, lr, optimizer):
    if optimizer == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9)
    elif optimizer == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        raise ValueError("Optimizer not recognized")


def train_iter(model, optimizer, x, y, device):
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, y)
    loss.backward()
    optimizer.step()
    accuracy = (output.argmax(1) == y).float().mean() * 100
    return loss.item(), accuracy.item()


def test_model(model, test_loader, device, attack=None):
    losses = []
    n_correct = 0
    n_samples = 0
    for i, (x, y) in enumerate(tqdm.tqdm(test_loader, leave=False)):
        x, y = x.to(device), y.to(device)
        if attack is not None:
            delta = attack.compute(x, y)
            x = torch.clamp(x + delta, 0, 1)
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        losses.append(loss.item())
        n_correct += (output.argmax(1) == y).sum().item()
        n_samples += y.size(0)
    return np.mean(losses), n_correct / n_samples * 100


def plot_results(losses_train, losses_test, acc_train, acc_test, args):
    plt.figure(figsize=(10, 5))

    best_ecoch = np.argmin(losses_test) 
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(losses_train))/len(losses_train)*(len(losses_test)-1),           
                        losses_train, label='Train')
    plt.plot(losses_test, label='Test')
    plt.plot([best_ecoch], [losses_test[best_ecoch]], color='red', label='Best test loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(len(acc_train))/len(acc_train)*(len(acc_test)-1),           
                        acc_train, label='Train')
    plt.plot(acc_test, label='Test')
    plt.plot([best_ecoch], [acc_test[best_ecoch]], color='red', label='Best test accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.path, 'results.png'))
    plt.close()


def visualize_results(x, y, model, device, args, attack=None):
    x, y = x.to(device), y.to(device)
    if attack is not None:
        delta = attack.compute(x, y)
        x = torch.clamp(x + delta, 0, 1)
    with torch.no_grad():
        y_pred = model(x).argmax(dim=1)
    #x, y, y_pred = x.cpu().numpy(), y.cpu().numpy(), y_pred.detach().cpu().numpy()
    x, y_pred = x.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
    f, ax = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(5*2, 5*1.3))
    for i in range(5):
        for j in range(5):
            img = x[i*5+j].transpose(1, 2, 0)
            img = np.maximum(0, np.minimum(img, 1))
            ax[i][j].imshow(img)
            title = ax[i][j].set_title("Pred: {}".format(cifar10_classes[int(y_pred[i*5+j])]))
            plt.setp(title, color=('g' if y_pred[i*5+j] == y[i*5+j] else 'r'))
            ax[i][j].set_axis_off()
        plt.tight_layout()
    if attack is not None:
        plt.savefig(os.path.join(args.path, f'examples_{attack.name}.png'))
    else:
        plt.savefig(os.path.join(args.path, 'examples.png'))
    plt.close()
