# from folder_name.file_name import class_name
from layers.fully_connected import FullyConnected
from layers.convolution import Convolution
from layers.pooling import Pooling
from layers.flatten import Flatten
from layers.activation import Elu, Softmax

from utilities.filereader import get_data
from utilities.model import Model

from utilities.utils import get_batches
from utilities.settings import get_models_path

from loss.losses import CategoricalCrossEntropy

import numpy as np
np.random.seed(0)
from os import path


if __name__ == '__main__':
    train_data, train_labels = get_data(num_samples=1000)
    test_data, test_labels = get_data(num_samples=1000, dataset="testing")

    train_data = train_data / 255
    test_data = test_data / 255

    print("Train data shape: {}, {}".format(train_data.shape, train_labels.shape))
    print("Test data shape: {}, {}".format(test_data.shape, test_labels.shape))

    model = Model(
        Convolution(filters=5, padding='same'),
        Elu(),
        Pooling(mode='max', kernel_shape=(2, 2), stride=2),
        Flatten(),
        FullyConnected(units=10),
        Softmax(),
        name='cnn5'
    )

    model.set_loss(CategoricalCrossEntropy)

    train_losses = []
    for epoch in range(2):
        print(f'Running Epoch: {epoch+1}')
        epoch_loss = 0
        batch_count = 0
        for i, (x_batch, y_batch) in enumerate(get_batches(train_data, train_labels.T)):
            batch_count += 1
            batch_preds = x_batch.copy()
            for num, layer in enumerate(model.model):
                batch_preds = layer.forward_propagate(batch_preds, save_cache=True)
            loss = CategoricalCrossEntropy.compute_loss(y_batch, batch_preds)
            epoch_loss += loss
            dA = CategoricalCrossEntropy.compute_derivative(y_batch, batch_preds)
            for layer in reversed(model.model):
                dA = layer.back_propagate(dA)
                if layer.has_weights():
                    layer.momentum()
                    layer.rmsprop()
            for layer in model.model:
                if layer.has_weights():
                    layer.apply_grads(optimization='adam', correct_bias=True, iter=i)
        train_losses.append(epoch_loss/batch_count)
        for layer in model.model:
            if layer.has_weights():
                layer.save_weights(path.join(get_models_path(), model.name))

    test_preds = model.predict(test_data)
    test_loss = CategoricalCrossEntropy.compute_loss(test_labels.T, test_preds)
    print('Testing accuracy = {}'.format(model.evaluate(test_data, test_labels)))
    print('Test loss = {}'.format(test_loss))

    # Plot training and test loss curves
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    plt.plot(range(1, 3), train_losses, label='Train Loss')
    plt.scatter([2], [test_loss], color='red', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Curves')
    plt.legend()
    import os
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/loss_curve_numpy.png')
    plt.show()