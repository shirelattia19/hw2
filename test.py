import os

import torch
import torchvision

from hw2 import layers, optimizers, training, answers
import torch
import unittest

from hw2.grad_compare import compare_layer_to_torch
import os
import numpy as np
import matplotlib.pyplot as plt
import unittest
import torch
import torchvision
import torchvision.transforms as tvtf
from cs236781.plot import plot_fit

test = unittest.TestCase()


N = 100
in_features = 200
num_classes = 10
eps = 1e-6
alpha = 0.1

def test_block_grad(block: layers.Layer, x, y=None, delta=1e-3):
    diffs = compare_layer_to_torch(block, x, y)

    # Assert diff values
    for diff in diffs:
        test.assertLess(diff, delta)


def test_part1_relu():
    lrelu = layers.LeakyReLU(alpha=alpha)
    x_test = torch.randn(N, in_features)

    # Test forward pass
    z = lrelu(x_test)
    test.assertSequenceEqual(z.shape, x_test.shape)
    test.assertTrue(torch.allclose(z, torch.nn.LeakyReLU(alpha)(x_test), atol=eps))

    # Test backward pass
    test_block_grad(lrelu, x_test)


def test_linear_layer():
    # Test Linear
    out_features = 1000
    fc = layers.Linear(in_features, out_features)
    x_test = torch.randn(N, in_features)

    # Test forward pass
    z = fc(x_test)
    test.assertSequenceEqual(z.shape, [N, out_features])
    torch_fc = torch.nn.Linear(in_features, out_features, bias=True)
    torch_fc.weight = torch.nn.Parameter(fc.w)
    torch_fc.bias = torch.nn.Parameter(fc.b)
    test.assertTrue(torch.allclose(torch_fc(x_test), z, atol=eps))

    # Test backward pass
    test_block_grad(fc, x_test)

    # Test second backward pass
    x_test = torch.randn(N, in_features)
    z = fc(x_test)
    z = fc(x_test)
    test_block_grad(fc, x_test)

def test_CE():
    # Test CrossEntropy
    cross_entropy = layers.CrossEntropyLoss()
    scores = torch.randn(N, num_classes)
    labels = torch.randint(low=0, high=num_classes, size=(N,), dtype=torch.long)

    # Test forward pass
    loss = cross_entropy(scores, labels)
    expected_loss = torch.nn.functional.cross_entropy(scores, labels)
    test.assertLess(torch.abs(expected_loss - loss).item(), 1e-5)
    print('loss=', loss.item())

    # Test backward pass
    test_block_grad(cross_entropy, scores, y=labels)

def test_sequencial():
    # Test Sequential
    # Let's create a long sequence of layers and see
    # whether we can compute end-to-end gradients of the whole thing.

    seq = layers.Sequential(
        layers.Linear(in_features, 100),
        layers.Linear(100, 200),
        layers.Linear(200, 100),
        layers.ReLU(),
        layers.Linear(100, 500),
        layers.LeakyReLU(alpha=0.01),
        layers.Linear(500, 200),
        layers.ReLU(),
        layers.Linear(200, 500),
        layers.LeakyReLU(alpha=0.1),
        layers.Linear(500, 1),
        layers.Sigmoid(),
    )
    x_test = torch.randn(N, in_features)

    # Test forward pass
    z = seq(x_test)
    test.assertSequenceEqual(z.shape, [N, 1])

    # Test backward pass
    test_block_grad(seq, x_test)

def test_mlp():
    in_features = 200
    num_classes = 10
    # Create an MLP model
    mlp = layers.MLP(in_features, num_classes, hidden_features=[100, 50, 100])
    print(mlp)
    # Test MLP architecture
    N = 100
    in_features = 10
    num_classes = 10
    for activation in ('relu', 'sigmoid'):
        mlp = layers.MLP(in_features, num_classes, hidden_features=[100, 50, 100], activation=activation)
        test.assertEqual(len(mlp.sequence), 7)

        num_linear = 0
        for b1, b2 in zip(mlp.sequence, mlp.sequence[1:]):
            if (str(b2).lower() == activation):
                test.assertTrue(str(b1).startswith('Linear'))
                num_linear += 1

        test.assertTrue(str(mlp.sequence[-1]).startswith('Linear'))
        test.assertEqual(num_linear, 3)

        # Test MLP gradients
        # Test forward pass
        x_test = torch.randn(N, in_features)
        labels = torch.randint(low=0, high=num_classes, size=(N,), dtype=torch.long)
        z = mlp(x_test)
        test.assertSequenceEqual(z.shape, [N, num_classes])

        # Create a sequence of MLPs and CE loss
        seq_mlp = layers.Sequential(mlp, layers.CrossEntropyLoss())
        loss = seq_mlp(x_test, y=labels)
        test.assertEqual(loss.dim(), 0)
        print(f'MLP loss={loss}, activation={activation}')

        # Test backward pass
        test_block_grad(seq_mlp, x_test, y=labels)


seed = 42
plt.rcParams.update({'font.size': 12})
test = unittest.TestCase()

def test_part2_trainer_layer():
    data_dir = os.path.expanduser('~/.pytorch-datasets')
    ds_train = torchvision.datasets.CIFAR10(root=data_dir, download=True, train=True, transform=tvtf.ToTensor())
    ds_test = torchvision.datasets.CIFAR10(root=data_dir, download=True, train=False, transform=tvtf.ToTensor())

    print(f'Train: {len(ds_train)} samples')
    print(f'Test: {len(ds_test)} samples')
    import hw2.layers as layers
    import hw2.answers as answers
    from torch.utils.data import DataLoader

    # Overfit to a very small dataset of 20 samples
    batch_size = 10
    max_batches = 2
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=False)

    # Get hyperparameters
    hp = answers.part2_overfit_hp()

    torch.manual_seed(seed)

    # Build a model and loss using our custom MLP and CE implementations
    model = layers.MLP(3 * 32 * 32, num_classes=10, hidden_features=[128] * 3, wstd=hp['wstd'])
    loss_fn = layers.CrossEntropyLoss()

    # Use our custom optimizer
    optimizer = optimizers.VanillaSGD(model.params(), learn_rate=hp['lr'], reg=hp['reg'])

    # Run training over small dataset multiple times
    trainer = training.LayerTrainer(model, loss_fn, optimizer)
    best_acc = 0
    for i in range(20):
        res = trainer.train_epoch(dl_train, max_batches=max_batches)
        best_acc = res.accuracy if res.accuracy > best_acc else best_acc

    test.assertGreaterEqual(best_acc, 98)


batch_size = 50
max_batches = 100
in_features = 3 * 32 * 32
num_classes = 10
data_dir = os.path.expanduser('~/.pytorch-datasets')
ds_train = torchvision.datasets.CIFAR10(root=data_dir, download=True, train=True, transform=tvtf.ToTensor())
ds_test = torchvision.datasets.CIFAR10(root=data_dir, download=True, train=False, transform=tvtf.ToTensor())
dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=False)
dl_test = torch.utils.data.DataLoader(ds_test, batch_size//2, shuffle=False)
# Define a function to train a model with our Trainer and various optimizers

def train_with_optimizer(opt_name, opt_class, fig):
    torch.manual_seed(seed)

    # Get hyperparameters
    hp = answers.part2_optim_hp()
    hidden_features = [128] * 5
    num_epochs = 10

    # Create model, loss and optimizer instances
    model = layers.MLP(in_features, num_classes, hidden_features, wstd=hp['wstd'])
    loss_fn = layers.CrossEntropyLoss()
    optimizer = opt_class(model.params(), learn_rate=hp[f'lr_{opt_name}'], reg=hp['reg'])

    # Train with the Trainer
    trainer = training.LayerTrainer(model, loss_fn, optimizer)
    fit_res = trainer.fit(dl_train, dl_test, num_epochs, max_batches=max_batches)

    fig, axes = plot_fit(fit_res, fig=fig, legend=opt_name)
    return fig

def test_part2_fit():
    # Define a larger part of the CIFAR-10 dataset (still not the whole thing)

    dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=False)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size // 2, shuffle=False)
    fig_optim = None
    fig_optim = train_with_optimizer('vanilla', optimizers.VanillaSGD, fig_optim)

def test_part2_rms():
    fig_optim = None
    fig_optim = train_with_optimizer('rmsprop', optimizers.RMSProp, fig_optim)
    fig_optim

def test_dropout():
    from hw2.grad_compare import compare_layer_to_torch

    # Check architecture of MLP with dropout layers
    mlp_dropout = layers.MLP(in_features, num_classes, [50] * 3, dropout=0.6)
    print(mlp_dropout)
    test.assertEqual(len(mlp_dropout.sequence), 10)
    for b1, b2 in zip(mlp_dropout.sequence, mlp_dropout.sequence[1:]):
        if str(b1).lower() == 'relu':
            test.assertTrue(str(b2).startswith('Dropout'))
    test.assertTrue(str(mlp_dropout.sequence[-1]).startswith('Linear'))

    # Test end-to-end gradient in train and test modes.
    print('Dropout, train mode')
    mlp_dropout.train(True)
    for diff in compare_layer_to_torch(mlp_dropout, torch.randn(500, in_features)):
        test.assertLess(diff, 1e-3)

    print('Dropout, test mode')
    mlp_dropout.train(False)
    for diff in compare_layer_to_torch(mlp_dropout, torch.randn(500, in_features)):
        test.assertLess(diff, 1e-3)

