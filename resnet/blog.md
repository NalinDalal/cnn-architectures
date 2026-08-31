# [Deep Residual Learning Explained: Implementing ResNet-18 in PyTorch]()

## Introduction

Deep learning models for computer vision have evolved rapidly over the past decade. Early convolutional neural networks such as AlexNet demonstrated that deep architectures could outperform traditional computer vision techniques. However, simply stacking more layers introduced a serious challenge: **very deep networks became difficult to train**.

In 2016, researchers from Microsoft Research introduced **Residual Networks (ResNet)** in the paper:

**Deep Residual Learning for Image Recognition — He et al.**

ResNet introduced a simple but powerful concept: **skip connections**, also known as **residual connections**. These connections allow information and gradients to flow directly through the network, enabling the successful training of extremely deep neural networks.

In this article we will:

* Understand why deep networks fail
* Learn how **residual learning** works
* Implement **ResNet-18 in PyTorch**
* Compare **Plain CNN vs ResNet**
* Analyze **training stability and accuracy**

---

# Why Deeper Networks Become Harder to Train

Intuitively, deeper networks should perform better because they can represent more complex functions.

However, in practice something unexpected happens:

As depth increases:

* training error increases
* optimization becomes unstable

This is known as the **degradation problem**.

Even when overfitting is not present, deeper networks can perform worse than shallower ones.

The core reason is related to **gradient propagation during backpropagation**.

---

# Vanishing Gradient Problem

During backpropagation, gradients flow from the output layer toward earlier layers.

In deep networks:

* gradients shrink exponentially
* earlier layers receive extremely small updates

This is called the **vanishing gradient problem**.

As a result:

* learning slows down
* optimization fails
* deeper networks become ineffective

ResNet was designed specifically to solve this issue.

---

# Residual Learning

The key idea behind ResNet is **residual learning**.

Instead of learning the direct mapping:

```
H(x)
```

the network learns the residual function:

```
F(x) = H(x) − x
```

The final output becomes:

```
y = F(x) + x
```

If the optimal mapping is close to identity, the network only needs to learn a small correction.

This greatly simplifies optimization.

---

# Residual Block

The **residual block** is the fundamental building unit of ResNet.

Inside a residual block:

```
Input
 ↓
Conv → BatchNorm → ReLU
 ↓
Conv → BatchNorm
 ↓
Add Skip Connection
 ↓
ReLU
```

The skip connection adds the input directly to the output of the convolution layers.

This allows the model to preserve information while learning useful transformations.

---

# Residual Block Diagram

```
        ┌───────────────────────┐
        │                       │
Input ──┤ Conv → BN → ReLU     │
        │        ↓              │
        │      Conv → BN        │
        │        ↓              │
        └─────── + ─────────────┘
                │
              ReLU
                │
              Output
```

The top branch is the **identity shortcut**.

---

# ResNet Architecture

ResNet models are defined by the number of layers.

| Model      | Layers |
| ---------- | ------ |
| ResNet-18  | 18     |
| ResNet-34  | 34     |
| ResNet-50  | 50     |
| ResNet-101 | 101    |
| ResNet-152 | 152    |

ResNet-152 achieved **3.57% top-5 error on ImageNet**, a major breakthrough at the time.

---

# Implementing ResNet-18 in PyTorch

### Residual Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(x)
        return F.relu(out)
```

---

# Plain CNN vs ResNet-18

To understand the impact of residual connections, we implemented two models.

### Plain CNN

```
Conv → ReLU → MaxPool
Conv → ReLU → MaxPool
Conv → ReLU → MaxPool
Flatten
Fully Connected
Softmax
```

### ResNet-18

```
Conv7×7
MaxPool
Residual Block ×2
Residual Block ×2
Residual Block ×2
Residual Block ×2
Global Average Pool
Fully Connected
```

Dataset used:

**CIFAR-10**

* 60,000 images
* 10 classes

---

# Training Script (PyTorch)

Below is a **clean training loop**.

```python
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True)

model = ResNet18()
device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4
)

for epoch in range(30):

    running_loss = 0

    for inputs, labels in trainloader:

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print("Epoch:", epoch, "Loss:", running_loss)
```

---

# Accuracy Comparison

After training on CIFAR-10:

| Model     | Test Accuracy |
| --------- | ------------- |
| Plain CNN | ~78%          |
| ResNet-18 | ~90%          |

ResNet significantly improves performance.

---

# Training Stability

Residual connections also improve **optimization stability**.

### Plain CNN Loss Curve

```
Loss
│\
│ \
│  \__
│      plateau
└────────────
```

### ResNet Loss Curve

```
Loss
│\
│ \
│  \
│   \
│    \__
└────────────
```

ResNet maintains **steady gradient flow**, leading to smoother convergence.

---

# Why ResNet Works So Well

Residual connections provide three key benefits.

### Better Gradient Flow

Gradients can propagate directly through identity connections.

### Identity Mapping

If deeper layers are unnecessary, the network learns:

```
F(x) = 0
```

which means:

```
y = x
```

Thus performance cannot degrade.

### Easier Optimization

Learning residual corrections is easier than learning the full mapping.

---

# Practical Tips for Training ResNet

When training residual networks:

* use **Batch Normalization**
* apply **data augmentation**
* use **SGD with momentum**
* schedule learning rate decay
* train on **GPU**

For CIFAR-10, ResNet-18 usually converges within **30–60 epochs**.

---

# Conclusion

ResNet fundamentally changed deep learning architecture design. By introducing **residual connections**, it solved the degradation problem and enabled extremely deep neural networks.

Today, residual connections are used in many modern architectures including:

* ResNeXt
* DenseNet
* EfficientNet
* Vision Transformers

Understanding ResNet is essential for anyone studying deep learning systems or building modern computer vision models.

---

# Project Code

```
init.py
```

Contains:

* ResNet-18 implementation
* CIFAR-10 training pipeline
* evaluation scripts

---

# References

ResNet Paper
[https://arxiv.org/pdf/1512.03385](https://arxiv.org/pdf/1512.03385)

Dive Into Deep Learning
[https://d2l.ai/chapter_convolutional-modern/resnet.html](https://d2l.ai/chapter_convolutional-modern/resnet.html)
