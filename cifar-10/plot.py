import matplotlib.pyplot as plt

def mean(li):
    return sum(li)/len(li)

def plot_results(trainAccuracy, testAccuracy, trainLoss, testLoss, key, trainEpoch, dropout, learning_rate, weight_decay, batch_size, xavier):
    plt.figure(1, figsize=(10,6))
    plt.plot(trainAccuracy, label='Train Accuracy', color='red', marker='o', linewidth=2)
    plt.plot(testAccuracy, label='Test Accuracy', color='blue', marker='x', linewidth=2)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title(f"Epoch Accuracy Plot\n Dropout: {dropout}| Learning Rate: {learning_rate}| Optimizers: {key} | Weight Decay:{weight_decay}| Batch Size: {batch_size}| Filter Weight(xavier_normal): {xavier}")
    plt.legend()
    plt.savefig(f'acc_{learning_rate}_{key}_{weight_decay}_{batch_size}_{xavier}.jpg', dpi=100, bbox_inches='tight')
    plt.figure(2, figsize=(10,6))
    plt.plot(trainLoss, label='Train loss', color='r', marker='o', linewidth=2)
    plt.plot(testLoss, label='Test loss', color='b', marker='x', linewidth=2)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title(f"Epoch Loss Plot\n Dropout: {dropout}| Learning Rate: {learning_rate}| Optimizers: {key} | Weight Decay: {weight_decay}| Batch Size: {batch_size}| Filter Weight(xavier_normal): {xavier}")
    plt.legend()
    plt.savefig(f'loss_{learning_rate}_{key}_{weight_decay}_{batch_size}_{xavier}.jpg', dpi=100, bbox_inches='tight')
    plt.figure(3, figsize=(12,4))
    plt.plot([mean(trainEpoch[i:i+500]) for i in range(len(trainEpoch)-50)])
    plt.xlabel('training batch-size')
    plt.ylabel('loss')
    plt.title(f"Performance Curve\n Dropout: {dropout}| Learning Rate: {learning_rate}| Optimizers: {key} | Weight Decay: {weight_decay}| Batch Size: {batch_size}| Filter Weight(xavier_normal): {xavier}")
    plt.savefig(f'per_{learning_rate}_{key}_{weight_decay}_{batch_size}_{xavier}.jpg',dpi=100, bbox_inches='tight')
