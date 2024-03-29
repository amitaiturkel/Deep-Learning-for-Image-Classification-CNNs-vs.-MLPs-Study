import os
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch
import torchvision
from tqdm import tqdm
from torchvision import transforms
import numpy as np
from xgboost import XGBClassifier

def get_loaders(path, transform, batch_size):
    """
    Get the data loaders for the train, validation and test sets.
    :param path: The path to the 'whichfaceisreal' directory.
    :param transform: The transform to apply to the images.
    :param batch_size: The batch size.
    :return: The train, validation and test data loaders.
    """
    train_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'train'), transform=transform)
    val_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'val'), transform=transform)
    test_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'test'), transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# Set the random seed for reproducibility
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.CenterCrop(64), transforms.ToTensor()])
batch_size = 32
path = r'C:\Users\home\Desktop\study\second year\Machine Learning Methods\ex4\whichfaceisreal'
train_loader, val_loader, test_loader = get_loaders(path, transform, batch_size)


# DATA LOADING
### DO NOT CHANGE THE CODE BELOW THIS LINE ###
train_data = []
train_labels = []
test_data = []
test_labels = []
with torch.no_grad():
    for (imgs, labels) in tqdm(train_loader, total=len(train_loader), desc='Train'):
        train_data.append(imgs)
        train_labels.append(labels)
    train_data = torch.cat(train_data, 0).cpu().numpy().reshape(len(train_loader.dataset), -1)
    train_labels = torch.cat(train_labels, 0).cpu().numpy()
    for (imgs, labels) in tqdm(test_loader, total=len(test_loader), desc='Test'):
        test_data.append(imgs)
        test_labels.append(labels)
    test_data = torch.cat(test_data, 0).cpu().numpy().reshape(len(test_loader.dataset), -1)
    test_labels = torch.cat(test_labels, 0).cpu().numpy()
### DO NOT CHANGE THE CODE ABOVE THIS LINE ###


### YOUR XGBOOST CODE GOES HERE ###
lr = [0.00001,0.0001,0.001,0.01,0.1]
scores = np.array([]) 
for learning_rate in lr:
    xgb_classifier = XGBClassifier(learning_rate =learning_rate)
    xgb_classifier.fit(train_data,train_labels)
    predictions = xgb_classifier.predict(test_data)
    score = np.mean(predictions == test_labels)
    scores = np.append(scores,score)
    print(score,learning_rate)
best_index = scores.argmax()
best_score = scores[best_index]
best_lr = lr[best_index]
min_index = scores.argmin()
scores[best_index] = -float('inf')
sec_best_index = scores.argmax()
sec_best_score = scores[sec_best_index]
sec_best_lr = lr[sec_best_index]
print(f"best Xg model is with test score of {best_score} and  learning rate of {best_lr}")
print(f"second best Xg model is with test score of {sec_best_score} and  learning rate of {sec_best_lr}")
print(f"worst Xg model is with test score of {scores[min_index]} and learning rate of {lr[min_index]}")

    
    

