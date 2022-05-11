import numpy as np 
from PIL import Image

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

IM_SIZE = 300
NUM_CL = 10

class FblClassifier(nn.Module):
        def __init__(self):
            super(FblClassifier, self).__init__()
            self.block1 = self.conv_block(c_in=3, c_out=256, dropout=0.1, kernel_size=5, stride=1, padding=2)
            self.block2 = self.conv_block(c_in=256, c_out=128, dropout=0.1, kernel_size=3, stride=1, padding=1)
            self.block3 = self.conv_block(c_in=128, c_out=64, dropout=0.1, kernel_size=3, stride=1, padding=1)
            self.lastcnn = nn.Conv2d(in_channels=64, out_channels=NUM_CL, kernel_size=75, stride=1, padding=0)
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        def forward(self, x):
            x = self.block1(x)
            x = self.maxpool(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.maxpool(x)
            x = self.lastcnn(x)
            return x
        def conv_block(self, c_in, c_out, dropout, **kwargs):
            seq_block = nn.Sequential(
                nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
                nn.BatchNorm2d(num_features=c_out),
                nn.ReLU(),
                nn.Dropout2d(p=dropout)
            )
            return seq_block

model = FblClassifier()
model.load_state_dict(torch.load('./fotmodel.pt', map_location=torch.device('cpu')))
model.eval()

image_transforms = transforms.Compose([
        transforms.Resize((IM_SIZE, IM_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])

idx2class = {0: 'Arsenal',
1: 'Barcelona',
2: 'Bayern',
3: 'Chelsea',
4: 'Juventus',
5: 'Liverpool',
6: 'ManchesterCity',
7: 'ManchesterUnited',
8: 'PSG',
9: 'Real'}

def predict(img):
    imgt = image_transforms(img)
    imgt = imgt.unsqueeze(0)

    with torch.no_grad():
        pred = model(imgt)
        _, pred_tag = torch.max(pred, dim = 1)

    return idx2class[int(pred_tag[0][0][0])]
    