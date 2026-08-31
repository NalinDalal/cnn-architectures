import torch
import torch.nn as nn
import torch.nn.functional as F

class modelCIFAR(nn.Module):
    def __init__(self, xavier=False, dropout=True):
        super(modelCIFAR, self).__init__()
        self.convLayer1 = nn.Conv2d(3, 32, 5)
        self.maxPool = nn.MaxPool2d(2, 2)
        self.convLayer2 = nn.Conv2d(32, 64, 5)
        self.drop1 = nn.Dropout(0.2, inplace=False) if dropout else nn.Identity()
        self.fullyc1 = nn.Linear(64 * 5 * 5, 200)
        if xavier:
            nn.init.xavier_normal_(self.fullyc1.weight)
        self.fullyc2 = nn.Linear(200, 100)
        self.fullyc3 = nn.Linear(100, 10)

    def forward(self, img):
        img = self.maxPool(self.drop1(F.relu(self.convLayer1(img))) )
        img = self.maxPool(self.drop1(F.relu(self.convLayer2(img))) )
        img = img.view(-1, 64 * 5 * 5)
        img = F.relu(self.fullyc1(img))
        img = F.relu(self.fullyc2(img))
        img = self.fullyc3(img)
        return img
