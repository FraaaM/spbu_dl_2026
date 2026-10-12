"""Общие функции для дз 1: базовая сеть, загрузка FashionMNIST, обучение и валидация."""

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import datasets

DATA_DIR = Path(__file__).resolve().parent / "data"

# среднее и std пикселей обучающей части FashionMNIST
FMNIST_MEAN, FMNIST_STD = 0.2860, 0.3530

FMNIST_CLASSES = [
    "T-Shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot",
]


class BaseNetwork(nn.Module):

    def __init__(self, act_fn, input_size=784, num_classes=10, hidden_sizes=[512, 256, 256, 128]):
        """
        Inputs:
            act_fn - Object of the activation function that should be used as non-linearity in the network.
            input_size - Size of the input images in pixels
            num_classes - Number of classes we want to predict
            hidden_sizes - A list of integers specifying the hidden layer sizes in the NN
        """
        super().__init__()

        # Create the network based on the specified hidden sizes
        layers = []
        layer_sizes = [input_size] + hidden_sizes
        for layer_index in range(1, len(layer_sizes)):
            layers += [nn.Linear(layer_sizes[layer_index-1], layer_sizes[layer_index]),
                       act_fn]
        layers += [nn.Linear(layer_sizes[-1], num_classes)]
        self.layers = nn.ModuleList(layers) # A module list registers a list of modules as submodules (e.g. for parameters)

        self.config = {"act_fn": act_fn.__class__.__name__, "input_size": input_size, "num_classes": num_classes, "hidden_sizes": hidden_sizes}

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for l in self.layers:
            x = l(x)
        return x


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_fashion_mnist(root=DATA_DIR):
    """Загружает FashionMNIST целиком в тензоры (так сильно быстрее, чем DataLoader с PIL).

    Возвращает x_train, y_train, x_test, y_test; картинки нормированы, форма (N, 784).
    """
    train = datasets.FashionMNIST(root, train=True, download=True)
    test = datasets.FashionMNIST(root, train=False, download=True)

    def prepare(ds):
        x = ds.data.float().div(255.0).sub(FMNIST_MEAN).div(FMNIST_STD)
        return x.view(len(ds), -1), ds.targets.clone()

    x_train, y_train = prepare(train)
    x_test, y_test = prepare(test)
    return x_train, y_train, x_test, y_test


def iterate_batches(x, y, batch_size, shuffle=True, generator=None):
    """Простой аналог DataLoader для данных, которые уже лежат в памяти."""
    n = len(x)
    idx = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        batch_idx = idx[start:start + batch_size]
        yield x[batch_idx], y[batch_idx]


def grad_norm(model):
    """Общая L2-норма градиентов по всем параметрам модели."""
    norms = [p.grad.detach().norm() for p in model.parameters() if p.grad is not None]
    if not norms:
        return 0.0
    return torch.stack(norms).norm().item()


def train_epoch(model, x, y, optimizer, loss_fn=None, batch_size=128, generator=None):
    """Одна эпоха обучения. Возвращает лоссы и нормы градиентов по батчам."""
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    model.train()
    losses, grad_norms = [], []
    for xb, yb in iterate_batches(x, y, batch_size, shuffle=True, generator=generator):
        loss = loss_fn(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        grad_norms.append(grad_norm(model))
        optimizer.step()
        losses.append(loss.item())
    return losses, grad_norms


@torch.no_grad()
def evaluate(model, x, y, batch_size=2048):
    """Средний кросс-энтропийный лосс и accuracy на выборке."""
    model.eval()
    total_loss, correct = 0.0, 0
    for xb, yb in iterate_batches(x, y, batch_size, shuffle=False):
        logits = model(xb)
        total_loss += nn.functional.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(1) == yb).sum().item()
    return total_loss / len(x), correct / len(x)
