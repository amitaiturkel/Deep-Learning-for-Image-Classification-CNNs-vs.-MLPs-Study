import os
from matplotlib import pyplot as plt
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch
import torchvision
from tqdm import tqdm
from torchvision import transforms
import numpy as np
from sklearn.linear_model import LogisticRegression

class ResNet18(nn.Module):
    def __init__(self, pretrained=False, probing=False):
        super(ResNet18, self).__init__()
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
            self.resnet18 = resnet18(weights=weights)
        else:
            self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
            self.resnet18 = resnet18()
        in_features_dim = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Identity()
        if probing:
            for name, param in self.resnet18.named_parameters():
                    param.requires_grad = False
        self.logistic_regression = nn.Linear(in_features_dim, 1)

    def forward(self, x):
        features = self.resnet18(x)
        ### YOUR CODE HERE ###
        x = features.view(features.size(0), -1)
        logits = self.logistic_regression(x)
        return logits


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

def compute_accuracy(model, data_loader, device):
    """
    Compute the accuracy of the model on the data in data_loader
    :param model: The model to evaluate.
    :param data_loader: The data loader.
    :param device: The device to run the evaluation on.
    :return: The accuracy of the model on the data in data_loader
    """

    model.eval()
    correct = 0
    total = len(data_loader.dataset)
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = (torch.sigmoid(outputs) > 0.5).long().squeeze(1)
            correct += (predictions == labels).sum().item()

    accuracy = correct / total
    return accuracy

def run_training_epoch(model, criterion, optimizer, train_loader, device):
    """
    Run a single training epoch
    :param model: The model to train
    :param criterion: The loss function
    :param optimizer: The optimizer
    :param train_loader: The data loader
    :param device: The device to run the training on
    :return: The average loss for the epoch.
    """
    model.train()
    total_loss = 0
    for (imgs, labels) in tqdm(train_loader, total=len(train_loader)):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels.float().view(-1, 1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    return average_loss



### UNCOMMENT THE FOLLOWING LINES TO TRAIN THE MODEL ###
def from_scratch():
    """
    Train the model from scratch and evaluate its performance.
    """
    model = ResNet18(pretrained=False, probing=False)
    models_acc = np.array([])
    models_acc_unreal = np.array([])
    lr = [0.00001,0.0001,0.001,0.01,0.1]

    for learning_rate in lr:
        model = ResNet18(pretrained=False, probing=False)
        transform = model.transform
        batch_size = 32
        num_of_epochs = 1
        path = 'whichfaceisreal' # For example '/cs/usr/username/whichfaceisreal/'
        train_loader, val_loader, test_loader = get_loaders(path, transform, batch_size)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        ### Define the loss function and the optimizer
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # Run a training epoch
        loss = run_training_epoch(model, criterion, optimizer, train_loader, device)
        # Compute the accuracy
        train_acc = compute_accuracy(model, train_loader, device)
        # Compute the validation accuracy
        val_acc = compute_accuracy(model, val_loader, device)
        print(f'Epoch 1, Loss: {loss:.4f}, Val accuracy: {val_acc:.4f}')
        # Compute the test accuracy
        test_acc = compute_accuracy(model, test_loader, device)
        models_acc =np.append(models_acc,test_acc)
        models_acc_unreal = np.append(models_acc_unreal,test_acc)
        name = f"From Scratch with learning rate of {learning_rate}"

    best_model_index = np.argmax(models_acc_unreal)
    worse_model_index = np.argmin(models_acc_unreal)
    models_acc_unreal[best_model_index] = - float('inf')
    second_best_index = np.argmax(models_acc_unreal)
    print(f"From Scratch the best model is with acc of {models_acc[best_model_index]} and learning rate {lr[best_model_index]}")
    print(f"From Scratch the second best model is with acc of {models_acc[second_best_index]} and learning rate {lr[second_best_index]}")
    print(f"From Scratch the worse model is with acc of {models_acc[worse_model_index]} and learning rate {lr[worse_model_index]}")
    print(models_acc)
    print("")


def Linear_probing():
    """
    Train the model with linear probing and evaluate its performance.
    """
    models_acc = np.array([])
    models_acc_unreal = np.array([])
    model = ResNet18(pretrained=True, probing=True)
    lr = [0.00001,0.0001,0.001,0.01,0.1]
    for learning_rate in lr:
        model = ResNet18(pretrained=True, probing=True)
        transform = model.transform
        batch_size = 32
        num_of_epochs = 1
        path = 'whichfaceisreal' # For example '/cs/usr/username/whichfaceisreal/'
        train_loader, val_loader, test_loader = get_loaders(path, transform, batch_size)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        ### Define the loss function and the optimizer
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        ### Train the model

        

        # Run a training epoch
        loss = run_training_epoch(model, criterion, optimizer, train_loader, device)
        # Compute the accuracy
        train_acc = compute_accuracy(model, train_loader, device)
        # Compute the validation accuracy
        val_acc = compute_accuracy(model, val_loader, device)
        print(f'Epoch 1, Loss: {loss:.4f}, Val accuracy: {val_acc:.4f}')

        # Compute the test accuracy
        test_acc = compute_accuracy(model, test_loader, device)
        models_acc =np.append(models_acc,test_acc)
        models_acc_unreal = np.append(models_acc_unreal,test_acc)


    best_model_index = np.argmax(models_acc_unreal)
    worse_model_index = np.argmin(models_acc_unreal)
    models_acc_unreal[best_model_index] = - float('inf')
    second_best_index = np.argmax(models_acc_unreal)
    print(f"Linear probing the best model is with acc of {models_acc[best_model_index]} and learning rate {lr[best_model_index]}")
    print(f"Linear probing the second best model is with acc of {models_acc[second_best_index]} and learning rate {lr[second_best_index]}")
    print(f"Linear probing the worse model is with acc of {models_acc[worse_model_index]} and learning rate {lr[worse_model_index]}")
    print(models_acc)
    print("")
    

def fine_tuning():
    """
    Fine-tune the pretrained model and evaluate its performance.
    """
    models_acc = np.array([])
    models_acc_unreal = np.array([])
    lr = [0.00001,0.0001,0.001,0.01,0.1]
    for learning_rate in lr:
        model = ResNet18(pretrained=True, probing=False)
        transform = model.transform
        batch_size = 32
        num_of_epochs = 1
        path = 'whichfaceisreal' # For example '/cs/usr/username/whichfaceisreal/'
        train_loader, val_loader, test_loader = get_loaders(path, transform, batch_size)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        ### Define the loss function and the optimizer
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        ### Train the model


        # Run a training epoch
        loss = run_training_epoch(model, criterion, optimizer, train_loader, device)
        # Compute the accuracy
        train_acc = compute_accuracy(model, train_loader, device)
        # Compute the validation accuracy
        val_acc = compute_accuracy(model, val_loader, device)
        print(f'Epoch 1, Loss: {loss:.4f}, Val accuracy: {val_acc:.4f}')

        # Compute the test accuracy
        test_acc = compute_accuracy(model, test_loader, device)
        models_acc =np.append(models_acc,test_acc)
        models_acc_unreal = np.append(models_acc_unreal,test_acc)
    best_model_index = np.argmax(models_acc_unreal)
    worse_model_index = np.argmin(models_acc_unreal)
    models_acc_unreal[best_model_index] = - float('inf')
    second_best_index = np.argmax(models_acc_unreal)
    print(f"Fine-tuning the best model is with acc of {models_acc[best_model_index]} and learning rate {lr[best_model_index]}")
    print(f"Fine-tuning the second best model is with acc of {models_acc[second_best_index]} and learning rate {lr[second_best_index]}")
    print(f"Fine-tuning the worse model is with acc of {models_acc[worse_model_index]} and learning rate {lr[worse_model_index]}")
    print(models_acc)
    print("")



def get_feature(model, data, device):
    """
    Extract features from the model.

    Args:
        model: The model.
        data: The data loader.
        device: The device to run the evaluation on.

    Returns:
        tuple: Extracted features and corresponding labels.
    """
    features = []
    labels = []
    
    model.eval()
    para = list(model.children())
    reversed_para = para[:-1]
    feature_extractor = nn.Sequential(*reversed_para)
    
    with torch.no_grad():
        for img, label in data:
            img = img.to(device)
            label = label.to(device)
            
            # Extract feature representation
            feature = feature_extractor(img)
            feature = feature.view(feature.size(0), -1)  # Flatten the features
            features.append(feature.cpu().numpy())  # Convert to NumPy array
            labels.append(label.cpu().numpy())  # Convert to NumPy array
            
    return np.concatenate(features), np.concatenate(labels)

    
def sklearn():
    """
    Train logistic regression using features extracted from the model.
    """
    model = ResNet18(pretrained=True, probing=False)
    transform = model.transform
    batch_size = 32
    num_of_epochs = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = 'whichfaceisreal' # For example '/cs/usr/username/whichfaceisreal/'
    train_loader, val_loader, test_loader = get_loaders(path, transform, batch_size)
    train_x ,train_y = get_feature(model,train_loader,device)
    test_x ,test_y = get_feature(model,test_loader,device)
    val_x ,val_y = get_feature(model,val_loader,device)
    LR_model = LogisticRegression(max_iter=num_of_epochs)
    LR_model.fit(train_x,train_y)
    train_acc = LR_model.score(train_x,train_y)
    test_acc = LR_model.score(test_x,test_y)  
    val_acc = LR_model.score(val_x,val_y)
    print(f" the model with sklearn linear regression as last model score \ntrain acc of {train_acc} and test acc of {test_acc} and val of {val_acc}")   


def mis_label():
    """
    Find mislabeled samples between the best and worst models.
    """
    # Create the "best model" with lr=0.01
    best_model = ResNet18(pretrained=True, probing=False)
    best_lr = 0.0001
    best_transform = best_model.transform
    best_batch_size = 32
    best_num_of_epochs = 1
    path = 'whichfaceisreal'  # For example '/cs/usr/username/whichfaceisreal/'
    best_train_loader, best_val_loader, best_test_loader = get_loaders(path, best_transform, best_batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model = best_model.to(device)

    # Define the loss function and the optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(best_model.parameters(), lr=best_lr)

    # Train the "best model"
    for epoch in range(best_num_of_epochs):
        # Run a training epoch
        loss = run_training_epoch(best_model, criterion, optimizer, best_train_loader, device)

        # Compute the accuracy
        train_acc = compute_accuracy(best_model, best_train_loader, device)

        # Compute the validation accuracy
        val_acc = compute_accuracy(best_model, best_val_loader, device)
        print(f'Epoch {epoch + 1}, Loss: {loss:.4f}, Val accuracy: {val_acc:.4f}')

    # Compute the test accuracy
    test_acc = compute_accuracy(best_model, best_test_loader, device)
    print(f"Best Model Test accuracy: {test_acc}")

    # Create the "worst model" with lr=0.1
    worst_model = ResNet18(pretrained=False, probing=False)
    worst_lr =  0.0001
    worst_transform = worst_model.transform
    worst_batch_size = 32
    worst_num_of_epochs = 1
    worst_train_loader, worst_val_loader, worst_test_loader = get_loaders(path, worst_transform, worst_batch_size)

    worst_model = worst_model.to(device)

    # Define the loss function and the optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(worst_model.parameters(), lr=worst_lr)

    # Train the "worst model"
    for epoch in range(worst_num_of_epochs):
        # Run a training epoch
        loss = run_training_epoch(worst_model, criterion, optimizer, worst_train_loader, device)

        # Compute the accuracy
        train_acc = compute_accuracy(worst_model, worst_train_loader, device)

        # Compute the validation accuracy
        val_acc = compute_accuracy(worst_model, worst_val_loader, device)
        print(f'Epoch {epoch + 1}, Loss: {loss:.4f}, Val accuracy: {val_acc:.4f}')

    # Compute the test accuracy
    test_acc = compute_accuracy(worst_model, worst_test_loader, device)
    print(f"Worst Model Test accuracy: {test_acc}")

    # Find 5 images that the "best model" predicted right and the "worst model" predicted wrong
    misclassified_samples = visualize_samples(best_model, worst_model, best_test_loader)

    # Plot the samples
    # Plot the samples
    plt.figure(figsize=(15, 5))
    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(misclassified_samples[i][0].squeeze().permute(1, 2, 0).numpy())
        plt.title(f"Best Model\nTrue Label: {misclassified_samples[i][1]}")
        plt.axis('off')

    plt.show()

def visualize_samples(model1, model2, loader, num_samples=5):
    model1.eval()
    model2.eval()
    misclassified_samples = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            outputs1 = model1(images)
            predictions1 = (torch.sigmoid(outputs1) > 0.5).int()  # Predictions for the entire batch

            outputs2 = model2(images)
            predictions2 = (torch.sigmoid(outputs2) > 0.5).int()  # Predictions for the entire batch

            # Iterate over predictions in the batch
            for i,pred1, pred2, label in zip(range(len(images)),predictions1, predictions2, labels):
                if pred1.item() == label.item() and pred2.item() != label.item():
                    misclassified_samples.append((images[i], label.item()))
                    if len(misclassified_samples) >= num_samples :
                        break

            if len(misclassified_samples) >= num_samples :
                break

    return misclassified_samples


