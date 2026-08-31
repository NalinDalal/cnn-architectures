import torch

def evaluate(model, data_loader, criterion, use_cuda):
    model.eval()
    correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in data_loader:
            if use_cuda and torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    return correct, total_loss
