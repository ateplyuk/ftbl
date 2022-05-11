import numpy as np 
import pandas as pd
from PIL import Image
import os
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import classification_report, confusion_matrix

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)
    return acc

EPOCHS = 5
LR = 0.0001
IM_SIZE = 300
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#TRAIN_DIR = '../input/football/GrayScaleTrain/'

def start_train(train_dir):
    image_transforms = transforms.Compose([
            transforms.Resize((IM_SIZE, IM_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])

    img_dataset = datasets.ImageFolder(root = train_dir, transform = image_transforms)

    idx2class = {v: k for k, v in img_dataset.class_to_idx.items()}
    NUM_CL = len(idx2class)

    img_dataset_size = len(img_dataset)
    img_dataset_indices = list(range(img_dataset_size))

    np.random.shuffle(img_dataset_indices)

    test_split_index = int(np.floor(0.3 * img_dataset_size))
    train_idx, test_idx = img_dataset_indices[test_split_index:], img_dataset_indices[:test_split_index]

    val_split_index = int(np.floor(0.3 * len(test_idx)))
    val_idx, test_idx = test_idx[val_split_index:], test_idx[:val_split_index]

    print(len(train_idx), len(val_idx), len(test_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(dataset=img_dataset, shuffle=False, batch_size=8, sampler=train_sampler)
    val_loader = DataLoader(dataset=img_dataset, shuffle=False, batch_size=1, sampler=val_sampler)

    single_batch = next(iter(train_loader))
    single_batch[0].shape
    
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
    model.to(DEVICE)
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }

    print("Begin training.")
    for e in tqdm(range(1, EPOCHS)):
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(DEVICE), y_train_batch.to(DEVICE)
            optimizer.zero_grad()
            y_train_pred = model(X_train_batch).squeeze()
            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)
            train_loss.backward()
            optimizer.step()
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()
        # VALIDATION
        with torch.no_grad():
            model.eval()
            val_epoch_loss = 0
            val_epoch_acc = 0
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(DEVICE), y_val_batch.to(DEVICE)
                y_val_pred = model(X_val_batch).squeeze()
                y_val_pred = torch.unsqueeze(y_val_pred, 0)
                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch)
                val_epoch_loss += train_loss.item()
                val_epoch_acc += train_acc.item()
        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc/len(val_loader))
        print(f'Epoch {e+0:02}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')


        torch.save(model.state_dict(), './fotmodel.pt')
        
# if __name__=='__main__':
#     main()
 

#start_train('../input/football/GrayScaleTrain/')    

# # Inference
# test_sampler = SubsetRandomSampler(test_idx)
# test_loader = DataLoader(dataset=img_dataset, shuffle=False, batch_size=1, sampler=test_sampler)

# y_pred_list = []
# y_true_list = []
# with torch.no_grad():
#     for x_batch, y_batch in tqdm(test_loader):
#         x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
#         y_test_pred = model(x_batch)
#         _, y_pred_tag = torch.max(y_test_pred, dim = 1)
#         y_pred_list.append(y_pred_tag.cpu().numpy())
#         y_true_list.append(y_batch.cpu().numpy())
        
# y_pred_list = [i[0][0][0] for i in y_pred_list]
# y_true_list = [i[0] for i in y_true_list]

# print(classification_report(y_true_list, y_pred_list))

# print(confusion_matrix(y_true_list, y_pred_list))