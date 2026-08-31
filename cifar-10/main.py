import torch
import torch.optim as optim
from hyperparams import get_hyperparams
from model import modelCIFAR
from data import load_data
from train import train
from evaluate import evaluate
from plot import plot_results

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

if __name__ == '__main__':
    params = get_hyperparams()
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    epoch_range = params['epoch_range']
    weight_decay = params['weight_decay']
    dropout = params['dropout']
    xavier = params['xavier']
    use_cuda = True
    key = 'SGD'
    traindataset, train_set, testdataset, test_set = load_data(batch_size)
    cifar = modelCIFAR(xavier=xavier, dropout=dropout)
    if use_cuda and torch.cuda.is_available():
        cifar.cuda()
    optimizer_dict = {'SGD': optim.SGD(cifar.parameters(), lr=learning_rate, momentum=0.9)}
    criterion = torch.nn.CrossEntropyLoss()
    for key, optimizer in optimizer_dict.items():
        trainAccuracy, testAccuracy, trainLoss, testLoss, trainEpoch = train(
            cifar, train_set, test_set, testdataset, optimizer, criterion,
            epoch_range, use_cuda, dropout, learning_rate, weight_decay, batch_size, xavier, key
        )
        plot_results(trainAccuracy, testAccuracy, trainLoss, testLoss, key, trainEpoch, dropout, learning_rate, weight_decay, batch_size, xavier)
        PATH = f'./cifar_net_{learning_rate}_{key}_{weight_decay}_{batch_size}_{xavier}.pth'
        torch.save(cifar.state_dict(), PATH)
        print(f"Model saved to {PATH}")
