import torch
import time
from evaluate import evaluate

def train(model, train_loader, test_loader, testdataset, optimizer, criterion, epoch_range, use_cuda, dropout, learning_rate, weight_decay, batch_size, xavier, key):
    trainLoss, testLoss, trainAccuracy, trainEpoch, testAccuracy = [], [], [], [], []
    for epoch in range(epoch_range):
        runningLoss = 0.0
        trAcc = 0.0
        totTrain = 0
        start_time = time.time()
        model.train()
        for i, data in enumerate(train_loader, 0):
            inputTrain, labelTrain = data
            if use_cuda and torch.cuda.is_available():
                inputTrain = inputTrain.cuda()
                labelTrain = labelTrain.cuda()
            optimizer.zero_grad()
            outputTrain = model(inputTrain)
            loss = criterion(outputTrain, labelTrain)
            loss.backward()
            optimizer.step()
            runningLoss += loss.item()
            trainEpoch.append(loss.item())
            totTrain += 1
            _, pred = torch.max(outputTrain, dim=1)
            correct_train = pred.eq(labelTrain.data.view_as(pred))
            accuracy_train = torch.mean(correct_train.type(torch.FloatTensor))
            trAcc += accuracy_train.item()
        corr, test_l = evaluate(model, test_loader, criterion, use_cuda)
        trainLoss.append(runningLoss/totTrain)
        testLoss.append(test_l/len(testdataset))
        trainAccuracy.append(trAcc/totTrain)
        testAccuracy.append(corr/len(testdataset))
        print(f"Epoch: {epoch+1}/{epoch_range} | Train loss: {runningLoss/totTrain:.3f} | Train Accuracy: {100*trAcc/totTrain:.3f} | Test loss: {test_l/len(testdataset):.3f} | Test Accuracy: {100*corr/len(testdataset):.3f} | Time/Epoch: {time.time() - start_time:.3f} sec|")
        runningLoss = 0.0
        trAcc = 0.0
    print('Finished Training')
    return trainAccuracy, testAccuracy, trainLoss, testLoss, trainEpoch
