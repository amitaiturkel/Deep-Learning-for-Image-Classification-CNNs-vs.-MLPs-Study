import time
import tabulate
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate
from helpers import *
import math
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

class SkipConnection(nn.Module):
    """A neural network module with skip connections.

    This module defines a feedforward neural network with skip connections added every 5 layers.
    """

    def __init__(self, input_dim, output_dim):
        """Initialize the SkipConnection module.

        Args:
            input_dim (int): The dimensionality of the input features.
            output_dim (int): The dimensionality of the output predictions.
        """
        super(SkipConnection, self).__init__()
        self.input_layer = nn.Linear(input_dim, 4)
        self.hidden_layers = nn.ModuleList([nn.Sequential(nn.Linear(4, 4), nn.ReLU()) for _ in range(99)])
        self.output_layer = nn.Linear(4, output_dim)

    def forward(self, x):
        """Perform a forward pass through the SkipConnection module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        out = torch.relu(self.input_layer(x))
        skip = out
        for i, layer in enumerate(self.hidden_layers):
            if i % 5 and i != 0:
                out = layer(out)
                out += skip
                skip = out
            else:
                out = layer(out)
        out = self.output_layer(out)
        return out


def train_model(train_data, val_data, test_data, model, lr=0.001, epochs=50, batch_size=256):
    """Train a neural network model.

    Args:
        train_data (pd.DataFrame): The training data.
        val_data (pd.DataFrame): The validation data.
        test_data (pd.DataFrame): The test data.
        model (nn.Module): The neural network model to train.
        lr (float, optional): The learning rate for optimization. Defaults to 0.001.
        epochs (int, optional): The number of epochs for training. Defaults to 50.
        batch_size (int, optional): The batch size for training. Defaults to 256.

    Returns:
        tuple: A tuple containing the trained model and lists of training, validation, and test metrics.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('Using device:', device)

    # Prepare datasets and data loaders
    trainset = torch.utils.data.TensorDataset(torch.tensor(train_data[['long', 'lat']].values).float(), torch.tensor(train_data['country'].values).long())
    valset = torch.utils.data.TensorDataset(torch.tensor(val_data[['long', 'lat']].values).float(), torch.tensor(val_data['country'].values).long())
    testset = torch.utils.data.TensorDataset(torch.tensor(test_data[['long', 'lat']].values).float(), torch.tensor(test_data['country'].values).long())

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1024, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_accs = []
    val_accs = []
    test_accs = []
    train_losses = []
    val_losses = []
    test_losses = []

    for ep in range(epochs):
        model.train()
        pred_correct = 0
        ep_loss = 0.
        for i, (inputs, labels) in enumerate(tqdm(trainloader)):
            # move the inputs and labels to the device
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the gradients
            optimizer.zero_grad()
            # forward pass
            outputs = model(inputs)
            # calculate the loss
            loss = criterion(outputs, labels)
            # backward pass
            loss.backward()
            
            # update the weights
            optimizer.step()

            pred_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            ep_loss += loss.item()

        train_accs.append(pred_correct / len(trainset))
        train_losses.append(ep_loss / len(trainloader))

        model.eval()
        with torch.no_grad():
            for loader, accs, losses in zip([valloader, testloader], [val_accs, test_accs], [val_losses, test_losses]):
                correct = 0
                total = 0
                ep_loss = 0.
                for inputs, labels in loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    # Forward pass (no backward pass in evaluation)
                    outputs = model(inputs)

                    # Calculate the loss
                    loss = criterion(outputs, labels)

                    ep_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                accs.append(correct / total)
                losses.append(ep_loss / len(loader))

        print('Epoch {:}, Train Acc: {:.3f}, Val Acc: {:.3f}, Test Acc: {:.3f}'.format(ep, train_accs[-1], val_accs[-1], test_accs[-1]))
    return model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses


def train_model_with_speed(train_data, val_data, test_data, model, lr=0.001, epochs=10, batch_size=32):
    """Train a neural network model and measure the training speed.

    Args:
        train_data (pd.DataFrame): The training data.
        val_data (pd.DataFrame): The validation data.
        test_data (pd.DataFrame): The test data.
        model (nn.Module): The neural network model to train.
        lr (float, optional): The learning rate for optimization. Defaults to 0.001.
        epochs (int, optional): The number of epochs for training. Defaults to 10.
        batch_size (int, optional): The batch size for training. Defaults to 32.

    Returns:
        tuple: A tuple containing the trained model and lists of training, validation, and test metrics.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('Using device:', device)

    # Prepare datasets and data loaders
    trainset = torch.utils.data.TensorDataset(torch.tensor(train_data[['long', 'lat']].values).float(), torch.tensor(train_data['country'].values).long())
    valset = torch.utils.data.TensorDataset(torch.tensor(val_data[['long', 'lat']].values).float(), torch.tensor(val_data['country'].values).long())
    testset = torch.utils.data.TensorDataset(torch.tensor(test_data[['long', 'lat']].values).float(), torch.tensor(test_data['country'].values).long())

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1024, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_accs = []
    val_accs = []
    test_accs = []
    train_losses = []
    val_losses = []
    test_losses = []
    speed_per_batch = []
    speed = 0

    for ep in range(epochs):
        model.train()
        pred_correct = 0
        ep_loss = 0.
        for i, (inputs, labels) in enumerate(tqdm(trainloader)):
            # move the inputs and labels to the device
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the gradients
            optimizer.zero_grad()
            # forward pass
            outputs = model(inputs)
            # calculate the loss
            loss = criterion(outputs, labels)
            # backward pass
            loss.backward()

            # update the weights
            optimizer.step()
            
            pred_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            ep_loss += loss.item()
            speed_per_batch.append(ep_loss)
        print(f" the speed for this epoch is {speed}")
        speed_per_batch.append(speed)
        train_accs.append(pred_correct / len(trainset))
        train_losses.append(ep_loss / len(trainloader))

        model.eval()
        with torch.no_grad():
            for loader, accs, losses in zip([valloader, testloader], [val_accs, test_accs], [val_losses, test_losses]):
                correct = 0
                total = 0
                ep_loss = 0.
                for inputs, labels in loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    # Forward pass (no backward pass in evaluation)
                    outputs = model(inputs)

                    # Calculate the loss
                    loss = criterion(outputs, labels)

                    ep_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                accs.append(correct / total)
                losses.append(ep_loss / len(loader))

        print('Epoch {:}, Train Acc: {:.3f}, Val Acc: {:.3f}, Test Acc: {:.3f}'.format(ep, train_accs[-1], val_accs[-1], test_accs[-1]))
    return model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses,speed_per_batch


def train_model_with_varied_lr(train_data, val_data, test_data, output_dim, learning_rates, epochs=50, batch_size=256):
    """Train a neural network model with varied learning rates and visualize the results.

    Args:
        train_data (pd.DataFrame): The training data.
        val_data (pd.DataFrame): The validation data.
        test_data (pd.DataFrame): The test data.
        output_dim (int): The dimensionality of the output predictions.
        learning_rates (list): A list of learning rates to try.
        epochs (int, optional): The number of epochs for training. Defaults to 50.
        batch_size (int, optional): The batch size for training. Defaults to 256.

    Returns:
        tuple: A tuple containing the trained model and a list of dictionaries, each containing learning rate and validation losses.
    """
    results = []

    for lr in learning_rates:
        # Define the model architecture
        model_para = [nn.Linear(2, 16), nn.ReLU(),  
             nn.Linear(16, 16), nn.ReLU(),  
             nn.Linear(16, 16), nn.ReLU(),  
             nn.Linear(16, 16), nn.ReLU(),  
             nn.Linear(16, 16), nn.ReLU(),  
             nn.Linear(16, 16), nn.ReLU(),  
             nn.Linear(16, output_dim)  
             ]
        
        model = nn.Sequential(*model_para)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        print(f'Using device: {device}, Learning Rate: {lr}')

        # Prepare datasets and data loaders
        trainset = torch.utils.data.TensorDataset(torch.tensor(train_data[['long', 'lat']].values).float(), torch.tensor(train_data['country'].values).long())
        valset = torch.utils.data.TensorDataset(torch.tensor(val_data[['long', 'lat']].values).float(), torch.tensor(val_data['country'].values).long())
        testset = torch.utils.data.TensorDataset(torch.tensor(test_data[['long', 'lat']].values).float(), torch.tensor(test_data['country'].values).long())

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        valloader = torch.utils.data.DataLoader(valset, batch_size=1024, shuffle=False, num_workers=0)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=0)

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_losses = []
        val_losses = []

        for ep in range(epochs):
            model.train()
            pred_correct = 0
            ep_loss = 0.
            for i, (inputs, labels) in enumerate(tqdm(trainloader)):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                pred_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
                ep_loss += loss.item()

            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                ep_loss_val = 0.
                for inputs, labels in valloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    ep_loss_val += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                val_losses.append(ep_loss_val / len(valloader))

            print(f'Epoch {ep}, Learning Rate: {lr}, Train Acc: {pred_correct / len(trainset):.3f}, Val Acc: {correct / total:.3f}')

        results.append({'learning_rate': lr, 'val_losses': val_losses})

    plt.figure(figsize=(10, 6))
    for result in results:
        # Use whole epochs on the x-axis
        plt.plot(range(1, epochs + 1), result['val_losses'], label=f'LR: {result["learning_rate"]:.5f}')

    # Set x-axis ticks to whole numbers only
    plt.xticks(range(1, epochs + 1))

    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Validation Loss Over Epochs for Different Learning Rates')
    plt.show()

    return model, results


def compare_two_models(model1, model2):
    """Compare the performance of two neural network models.

    Args:
        model1 (nn.Module): The first neural network model.
        model2 (nn.Module): The second neural network model.
    """
    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')

    # Train and visualize model1
    model1, train_accs1, val_accs1, test_accs1, train_losses1, val_losses1, test_losses1 = train_model(train_data, val_data, test_data, model1)
    plt.figure()
    plt.plot(train_losses1, label='Train', color='red')
    plt.plot(val_losses1, label='Val', color='blue')
    plt.plot(test_losses1, label='Test', color='green')
    plt.title('Losses model without batch norma')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(train_accs1, label='Train', color='red')
    plt.plot(val_accs1, label='Val', color='blue')
    plt.plot(test_accs1, label='Test', color='green')
    plt.title('Accs without batch norma')
    plt.legend()
    plt.show()
    plot_decision_boundaries(model1, test_data[['long', 'lat']].values, test_data['country'].values, 'Decision Boundaries', implicit_repr=False)

    # Train and visualize model2
    model2, train_accs2, val_accs2, test_accs2, train_losses2, val_losses2, test_losses2 = train_model(train_data, val_data, test_data, model2)
    plt.figure()
    plt.plot(train_losses2, label='Train', color='red')
    plt.plot(val_losses2, label='Val', color='blue')
    plt.plot(test_losses2, label='Test', color='green')
    plt.title('Losses with batch norma')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(train_accs2, label='Train', color='red')
    plt.plot(val_accs2, label='Val', color='blue')
    plt.plot(test_accs2, label='Test', color='green')
    plt.title('Accs with batch norma')
    plt.legend()
    plt.show()

    plot_decision_boundaries(model2, test_data[['long', 'lat']].values, test_data['country'].values, 'Decision Boundaries', implicit_repr=False)



def Q_6_1_2():
    """
    This function performs various experiments with neural network models.

    It starts by training models with different learning rates and visualizes the validation losses
    at specific epochs.

    Then, it compares two neural network models with and without batch normalization.

    Finally, it trains multiple models with different batch sizes and epochs, and plots the training losses
    for each model.

    Returns:
        None
    """
    learning_rates = [1 , 0.01,0.001, 0.00001]
    varied_epochs = [1, 5, 10, 20, 50, 100]
    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')
    output_dim = len(train_data['country'].unique())
    
    train_model_with_varied_lr(train_data, val_data, test_data,output_dim ,learning_rates)
    model_para = [nn.Linear(2, 16), nn.ReLU(),  # hidden layer 1
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 2
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 3
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 4
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 5
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 6
             nn.Linear(16, output_dim)  # output layer
             ]
    model = nn.Sequential(*model_para)
    model, train_accs1, val_accs1, test_accs1, train_losses1, val_losses1, test_losses1 = train_model(train_data, val_data, test_data, model,epochs=100)
    lose_per_epoch = []
    for epoch in varied_epochs:
        lose_per_epoch.append(val_losses1[epoch-1])
    plt.plot(varied_epochs, lose_per_epoch, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss at Specific Epochs')
    plt.show()

    model_para = [nn.Linear(2, 16), nn.ReLU(),  # hidden layer 1
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 2
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 3
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 4
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 5
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 6
             nn.Linear(16, output_dim)  # output layer
             ]

    model = nn.Sequential(*model_para)
    batch_model_para = [
    nn.Linear(2, 16),nn.BatchNorm1d(16),nn.ReLU(),nn.Linear(16, 16),nn.BatchNorm1d(16),nn.ReLU(),nn.Linear(16, 16),
    nn.BatchNorm1d(16),nn.ReLU(),
    nn.Linear(16, 16),nn.BatchNorm1d(16),
    nn.ReLU(),nn.Linear(16, 16),
    nn.BatchNorm1d(16),nn.ReLU(),
    nn.Linear(16, 16),nn.BatchNorm1d(16),
    nn.ReLU(),nn.Linear(16, output_dim) 
    ]
    batch_model = nn.Sequential(*batch_model_para)
    compare_two_models(model,batch_model)
    batch_size_and_epochs = [(1, 1), (16, 10), (128, 50), (1024, 50)]
    
    # Plotting setup
    plt.figure(figsize=(10, 6))
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss for Different Models')
    plt.figure(figsize=(10, 6))
    plt.xlabel('batch_sizes')
    plt.ylabel('Loss')
    plt.title('Training Loss for Different Models per batch size')
    

    # Loop through batch_size_and_epochs to train models, plot losses, and collect results
    results_table = []
    for batch_size, epochs in batch_size_and_epochs:
        model_para = [nn.Linear(2, 16), nn.ReLU(),  # hidden layer 1
                    nn.Linear(16, 16), nn.ReLU(),  # hidden layer 2
                    nn.Linear(16, 16), nn.ReLU(),  # hidden layer 3
                    nn.Linear(16, 16), nn.ReLU(),  # hidden layer 4
                    nn.Linear(16, 16), nn.ReLU(),  # hidden layer 5
                    nn.Linear(16, 16), nn.ReLU(),  # hidden layer 6
                    nn.Linear(16, output_dim)  # output layer
                    ]
        model = nn.Sequential(*model_para)
        model_name = f'Model_Batch_{batch_size}_Epochs_{epochs}'

        
        model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, speed = train_model_with_speed(
        train_data, val_data, test_data, model, lr=0.001, epochs=epochs, batch_size=batch_size)
        plt.plot(range(1, len(speed) + 1), speed, label=model_name)

        # Plotting the training losses
        plt.plot(range(1, epochs + 1), train_losses, label=model_name)
        print(model_name, "Train losses:", train_losses)

        # Collect results for each model
        results_table.append({
            'Batch Size': batch_size,
            'Epochs': epochs,
            'Model Name': model_name,
            'Train Accuracy': train_accs[-1],
            'Test Accuracy': test_accs[-1],
            'Val Accuracy': val_accs[-1]

            
        })

    # Display the plot with legends
    plt.legend()
    plt.show()

    # Display the results in a pretty table
    headers = results_table[0].keys()
    table_data = [[result[header] for header in headers] for result in results_table]
    table_str = tabulate(table_data, headers=headers, tablefmt='pretty')
    print(table_str)
    # Display the plot with legends
    # Display the results in a pretty table
    headers = results_table[0].keys()
    table_data = [[result[header] for header in headers] for result in results_table]
    table_str = tabulate(table_data, headers=headers, tablefmt='pretty')
    print(table_str)
        
def Q_6_2():
    """
    This function compares the performance of several neural network models with different architectures.

    It trains multiple models with varying architectures, batch sizes, and epochs, and then visualizes
    the training, validation, and test losses for each model.

    Returns:
        None
    """
    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')
    output_dim = len(train_data['country'].unique())
    model1_para = [nn.Linear(2, 16), nn.ReLU(),  # hidden layer 1
            # hidden layer 6
            nn.Linear(16, output_dim)  # output layer
            ]
    model2_para = [nn.Linear(2, 16), nn.ReLU(),  # hidden layer 1
            nn.Linear(16, 16), nn.ReLU(),  # hidden layer 2
            nn.Linear(16, output_dim)  # output layer
            ]
    model3_para = [nn.Linear(2, 16), nn.ReLU(),  # hidden layer 1
            nn.Linear(16, 16), nn.ReLU(),  # hidden layer 2
            nn.Linear(16, 16), nn.ReLU(),  # hidden layer 3
            nn.Linear(16, 16), nn.ReLU(),  # hidden layer 4
            nn.Linear(16, 16), nn.ReLU(),  # hidden layer 5
            nn.Linear(16, 16), nn.ReLU(),  # hidden layer 6
            nn.Linear(16, output_dim)  # output layer
            ]
    model4_para = [nn.Linear(2, 16), nn.ReLU(),  # hidden layer 1
    nn.Linear(16, 16), nn.ReLU(),          # hidden layer 2
    nn.Linear(16, 16), nn.ReLU(),          # hidden layer 3
    nn.Linear(16, 16), nn.ReLU(),          # hidden layer 4
    nn.Linear(16, 16), nn.ReLU(),          # hidden layer 5
    nn.Linear(16, 16), nn.ReLU(),          # hidden layer 6
    nn.Linear(16, 16), nn.ReLU(),          # hidden layer 7
    nn.Linear(16, 16), nn.ReLU(),          # hidden layer 8
    nn.Linear(16, 16), nn.ReLU(),          # hidden layer 9
    nn.Linear(16, 16), nn.ReLU(),          # hidden layer 10
    nn.Linear(16, output_dim)]              # output layer
    model5_para = [
    nn.Linear(2, 8), nn.ReLU(),           #hidden layer 1
    nn.Linear(8, 8), nn.ReLU(),           # hidden layer 2
    nn.Linear(8, 8), nn.ReLU(),           # hidden layer 3
    nn.Linear(8, 8), nn.ReLU(),           # hidden layer 4
    nn.Linear(8, 8), nn.ReLU(),           # hidden layer 5
    nn.Linear(8, 8), nn.ReLU(),           # hidden layer 6
    nn.Linear(8, output_dim)]               # output layer
    model6_para = [
    nn.Linear(2, 32), nn.ReLU(),  # hidden layer 1
    nn.Linear(32, 32), nn.ReLU(),          # hidden layer 2
    nn.Linear(32, 32), nn.ReLU(),          # hidden layer 3
    nn.Linear(32, 32), nn.ReLU(),          # hidden layer 4
    nn.Linear(32, 32), nn.ReLU(),          # hidden layer 5
    nn.Linear(32, 32), nn.ReLU(),          # hidden layer 6
    nn.Linear(32, output_dim)              # output layer
]
    model7_para = [
    nn.Linear(2, 64), nn.ReLU(),  # hidden layer 1
    nn.Linear(64, 64), nn.ReLU(),          # hidden layer 2
    nn.Linear(64, 64), nn.ReLU(),          # hidden layer 3
    nn.Linear(64, 64), nn.ReLU(),          # hidden layer 4
    nn.Linear(64, 64), nn.ReLU(),          # hidden layer 5
    nn.Linear(64, 64), nn.ReLU(),          # hidden layer 6
    nn.Linear(64, output_dim)              # output layer
]
    models = []
    model1 = nn.Sequential(*model1_para)
    model2 = nn.Sequential(*model2_para)
    model3 = nn.Sequential(*model3_para)
    model4 = nn.Sequential(*model4_para)
    model5 = nn.Sequential(*model5_para)
    model6 = nn.Sequential(*model6_para)
    model7 = nn.Sequential(*model7_para)
    models.extend([model1, model2, model3, model4, model5, model6, model7])
    models_acc_val = [0] * len(models)
    models_acc_test = [0] * len(models)
    models_acc_train = [0] * len(models)
    models_loss_val = [0] * len(models)
    models_loss_test = [0] * len(models)
    models_loss_train = [0] * len(models)
    models_names = ["(1,16) ","(2,16)","(6,16)","(10,16)","(6,8)","(6,32)","(6,64)"]
    diffrent_batch_size = [256,16,256,1024,1024,16,1024]
    diffrent_epoch = [29,46,86,73,83,100,100]
    learning_rates = [0.01,0.001,0.01,0.01,0.01,0.001,0.001]
    models_names = [f"(1,16) lr ={learning_rates[0]} epoch = {diffrent_epoch[0]}, diffrent_batch_size = {diffrent_batch_size[0]} ",
                    f"(2,16) lr ={learning_rates[1]} epoch = {diffrent_epoch[1]}, diffrent_batch_size = {diffrent_batch_size[1]}",
                    f"(6,16) lr ={learning_rates[2]} epoch = {diffrent_epoch[2]}, diffrent_batch_size = {diffrent_batch_size[2]}",
                    f"(10,16)lr ={learning_rates[3]} epoch = {diffrent_epoch[3]}, diffrent_batch_size = {diffrent_batch_size[3]}",
                    f"(6,8) lr ={learning_rates[4]} epoch = {diffrent_epoch[4]}, diffrent_batch_size = {diffrent_batch_size[4]}",
                    f"(6,32) lr ={learning_rates[5]} epoch = {diffrent_epoch[5]}, diffrent_batch_size = {diffrent_batch_size[5]}"
                    ,f"(6,64) lr ={learning_rates[6]} epoch = {diffrent_epoch[6]}, diffrent_batch_size = {diffrent_batch_size[6]}"]
    
    for i in range(len(models)):
        models[i], models_acc_train[i], models_acc_val[i],models_acc_test[i] , models_loss_train[i], models_loss_val[i],models_loss_test[i]= train_model(train_data, val_data, test_data, models[i],
                                                               batch_size=diffrent_batch_size[i],lr =learning_rates[i],
                                                               epochs=diffrent_epoch[i])

        print(f"model {models_names[i]} with {diffrent_epoch[i]} epoches and {diffrent_batch_size[i]} batch size with {models_acc_val[i][-1]} acc val")
    max_val_accs_index = -1
    best_model = None
    max_val_accs = -1
    for i in range(len(models)):
        if models_acc_val[i][-1] >  max_val_accs:
            max_val_accs = models_acc_val[i][-1]
            max_val_accs_index = i
            best_model = models[i]
            
    print(f"the best model is with acc of {max_val_accs} and its model{max_val_accs_index +1}" )
    #so we can see that model 7 is the best within 10 epoch with val_acc of 0.9183566564394223
    i = max_val_accs_index
    best_train_accs, best_val_accs, best_test_accs, best_train_losses, best_val_losses, best_test_losses = models_acc_train[i], models_acc_val[i],models_acc_test[i],models_loss_train[i], models_loss_val[i],models_loss_test[i]
    epochs = np.arange(1, len(best_train_accs) + 1)
    # Loss Plots
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, best_train_losses, label='Training Loss')
    plt.plot(epochs, best_val_losses, label='Validation Loss')
    plt.plot(epochs, best_test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training, Validation, and Test Losses Over Epochs for model {models_names[i]}')
    plt.legend()
    plt.show()
    plot_decision_boundaries(best_model,test_data[['long', 'lat']].values, test_data['country'].values,title= "best model")
    min_val_accs_index = float('inf')
    min_val_accs = float('inf')
    worse_model = None
   
    for i in range(len(models)):
        if min_val_accs >  models_acc_val[i][-1]:
            min_val_accs = models_acc_val[i][-1]
            min_val_accs_index = i
            worse_model = models[i]
            
    print(f" the worst model is with val acc of {min_val_accs} is model{min_val_accs_index +1}" )
    i = min_val_accs_index
    worse_train_accs, worse_val_accs, worse_test_accs, worse_train_losses, worse_val_losses, worse_test_losses = models_acc_train[i], models_acc_val[i],models_acc_test[i],models_loss_train[i], models_loss_val[i],models_loss_test[i]
    epochs = np.arange(1, len(worse_train_accs) + 1)
    # Loss Plots
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, worse_train_losses, label='Training Loss')
    plt.plot(epochs, worse_val_losses, label='Validation Loss')
    plt.plot(epochs, worse_test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training, Validation, and Test Losses Over Epochs of worse_model {models_names[i]}')
    plt.legend()
    plt.show()
    plot_decision_boundaries(worse_model,test_data[['long', 'lat']].values, test_data['country'].values,title= "worse_model")


    # #so we can see that model 1 is the worse within 10 epoch with val_acc of 0.9183566564394223

def Q_6_2_3():
    """
    This function explores the relationship between the number of hidden layers in a neural network
    and its performance.

    It trains models with different numbers of hidden layers and records their training, validation,
    and test accuracies.

    Returns:
        None
    """
    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')

    output_dim = len(train_data['country'].unique())
    model1_para = [nn.Linear(2, 16), nn.ReLU(),  # hidden layer 1
               # hidden layer 6
             nn.Linear(16, output_dim)  # output layer
             ]
    model2_para = [nn.Linear(2, 16), nn.ReLU(),  # hidden layer 1
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 2
             nn.Linear(16, output_dim)  # output layer
             ]
    model3_para = [nn.Linear(2, 16), nn.ReLU(),  # hidden layer 1
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 2
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 3
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 4
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 5
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 6
             nn.Linear(16, output_dim)  # output layer
             ]
    model4_para = [nn.Linear(2, 16), nn.ReLU(),  # hidden layer 1
    nn.Linear(16, 16), nn.ReLU(),          # hidden layer 2
    nn.Linear(16, 16), nn.ReLU(),          # hidden layer 3
    nn.Linear(16, 16), nn.ReLU(),          # hidden layer 4
    nn.Linear(16, 16), nn.ReLU(),          # hidden layer 5
    nn.Linear(16, 16), nn.ReLU(),          # hidden layer 6
    nn.Linear(16, 16), nn.ReLU(),          # hidden layer 7
    nn.Linear(16, 16), nn.ReLU(),          # hidden layer 8
    nn.Linear(16, 16), nn.ReLU(),          # hidden layer 9
    nn.Linear(16, 16), nn.ReLU(),          # hidden layer 10
    nn.Linear(16, output_dim)] 

    # Define the list of models with varying numbers of hidden layers
    model_params_list = [model1_para,model2_para,model3_para,model4_para]

    # Train each model and record accuracies
    num_hidden_layers_list = []
    train_acc_list = []
    val_acc_list = []
    test_acc_list = []
    models_names = ["(1,16) ","(2,16)","(6,16)","(10,16)","(6,8)","(6,32)","(6,64)"]
    diffrent_batch_size = [256,16,256,1024,1024,16,1024]
    diffrent_epoch = [29,46,86,73,83,100,100]
    learning_rates = [0.01,0.001,0.01,0.01,0.01,0.001,0.001]

    for i, model_params in enumerate(model_params_list):
        num_hidden_layers = len([layer for layer in model_params if isinstance(layer, nn.Linear)]) - 1
        num_hidden_layers_list.append(num_hidden_layers)

        # Create model
        model = nn.Sequential(*model_params)

        # Train the model
        _, train_accs, val_accs, test_accs, _, _, _ = train_model(train_data, val_data, test_data, model, lr=learning_rates[i], epochs=diffrent_epoch[i], batch_size=diffrent_batch_size[i])

        # Record accuracies
        train_acc_list.append(train_accs[-1])
        val_acc_list.append(val_accs[-1])
        test_acc_list.append(test_accs[-1])

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(num_hidden_layers_list, train_acc_list, label='Training Accuracy', marker='o')
    plt.plot(num_hidden_layers_list, val_acc_list, label='Validation Accuracy', marker='o')
    plt.plot(num_hidden_layers_list, test_acc_list, label='Test Accuracy', marker='o')
    plt.xlabel('Number of Hidden Layers')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Number of Hidden Layers')
    plt.legend()
    plt.show()


def Q_6_2_4():
    """
    Trains multiple neural network models with varying architectures and plots the training, validation, and test accuracies.
    """
    
    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')

    output_dim = len(train_data['country'].unique())
    # para models with depth 6
    model3_para = [nn.Linear(2, 16), nn.ReLU(),  # hidden layer 1
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 2
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 3
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 4
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 5
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 6
             nn.Linear(16, output_dim)  # output layer
             ]
    model5_para = [
    nn.Linear(2, 8), nn.ReLU(),           #hidden layer 1
    nn.Linear(8, 8), nn.ReLU(),           # hidden layer 2
    nn.Linear(8, 8), nn.ReLU(),           # hidden layer 3
    nn.Linear(8, 8), nn.ReLU(),           # hidden layer 4
    nn.Linear(8, 8), nn.ReLU(),           # hidden layer 5
    nn.Linear(8, 8), nn.ReLU(),           # hidden layer 6
    nn.Linear(8, output_dim)]               # output layer
    model6_para = [
    nn.Linear(2, 32), nn.ReLU(),  # hidden layer 1
    nn.Linear(32, 32), nn.ReLU(),          # hidden layer 2
    nn.Linear(32, 32), nn.ReLU(),          # hidden layer 3
    nn.Linear(32, 32), nn.ReLU(),          # hidden layer 4
    nn.Linear(32, 32), nn.ReLU(),          # hidden layer 5
    nn.Linear(32, 32), nn.ReLU(),          # hidden layer 6
    nn.Linear(32, output_dim)              # output layer
]
    model7_para = [
    nn.Linear(2, 64), nn.ReLU(),  # hidden layer 1
    nn.Linear(64, 64), nn.ReLU(),          # hidden layer 2
    nn.Linear(64, 64), nn.ReLU(),          # hidden layer 3
    nn.Linear(64, 64), nn.ReLU(),          # hidden layer 4
    nn.Linear(64, 64), nn.ReLU(),          # hidden layer 5
    nn.Linear(64, 64), nn.ReLU(),          # hidden layer 6
    nn.Linear(64, output_dim)              # output layer
]

    # Define the list of models with varying numbers of hidden layers
    model_params_list = [model3_para,model5_para,model6_para,model7_para
    ]

    # Train each model and record accuracies
    num_width_layers_list = [16,8,32,64]
    train_acc_list = []
    val_acc_list = []
    test_acc_list = []
    models_names = ["(1,16) ","(2,16)","(6,16)","(10,16)","(6,8)","(6,32)","(6,64)"]
    diffrent_batch_size = [256,16,256,1024,1024,16,1024]
    diffrent_epoch = [29,46,86,73,83,100,99]
    learning_rates = [0.01,0.001,0.01,0.01,0.01,0.001,0.001]
    real_model = [2,4,5,6]
    

    for i, model_params in enumerate(model_params_list):

        # Create model
        model = nn.Sequential(*model_params)

        # Train the model
        _, train_accs, val_accs, test_accs, _, _, _ = train_model(train_data, val_data, test_data, model, lr=learning_rates[real_model[i]], epochs=diffrent_epoch[real_model[i]], batch_size=diffrent_batch_size[real_model[i]])

        # Record accuracies
        train_acc_list.append(train_accs[-1])
        val_acc_list.append(val_accs[-1])
        test_acc_list.append(test_accs[-1])

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(num_width_layers_list, train_acc_list, label='Training Accuracy', marker='o')
    plt.plot(num_width_layers_list, val_acc_list, label='Validation Accuracy', marker='o')
    plt.plot(num_width_layers_list, test_acc_list, label='Test Accuracy', marker='o')
    plt.xlabel('Width of Hidden Layers')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Width of Hidden Layers')
    plt.legend()
    plt.show()


def plot_average_gradient_magnitudes(gradient_magnitudes, layers_to_plot):
    """
    Plots the average gradient magnitudes for specific layers of a neural network model over different epochs.

    Parameters:
        gradient_magnitudes (dict): Dictionary containing average gradient magnitudes for each layer.
        layers_to_plot (list): List of layer indices to plot the gradient magnitudes for.
    """
    plt.figure(figsize=(10, 6))
    for layer_idx in layers_to_plot:
        layer_grad_mags = gradient_magnitudes[layer_idx]
        plt.plot(range(len(layer_grad_mags)), layer_grad_mags, label=f'Layer {layer_idx}')

    plt.xlabel('Epoch')
    plt.ylabel('Gradient Magnitude')
    plt.legend()
    plt.show()



def train_model_with_average_gradient_magnitudes(train_data, val_data, test_data, model, lr=0.001, epochs=10, batch_size=32, layers_to_plot=None):
    """
    Trains a neural network model while also computing and plotting average gradient magnitudes for specific layers.

    Parameters:
        train_data (DataFrame): Training data.
        val_data (DataFrame): Validation data.
        test_data (DataFrame): Test data.
        model (torch.nn.Module): Neural network model to train.
        lr (float): Learning rate for optimization (default is 0.001).
        epochs (int): Number of epochs for training (default is 10).
        batch_size (int): Batch size for training (default is 32).
        layers_to_plot (list): List of layer indices to plot the gradient magnitudes for.

    Returns:
        model (torch.nn.Module): Trained neural network model.
        train_accs (list): List of training accuracies.
        val_accs (list): List of validation accuracies.
        test_accs (list): List of test accuracies.
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
        test_losses (list): List of test losses.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('Using device:', device)

    trainset = torch.utils.data.TensorDataset(torch.tensor(train_data[['long', 'lat']].values).float(), torch.tensor(train_data['country'].values).long())
    valset = torch.utils.data.TensorDataset(torch.tensor(val_data[['long', 'lat']].values).float(), torch.tensor(val_data['country'].values).long())
    testset = torch.utils.data.TensorDataset(torch.tensor(test_data[['long', 'lat']].values).float(), torch.tensor(test_data['country'].values).long())

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1024, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_accs = []
    val_accs = []
    test_accs = []
    train_losses = []
    val_losses = []
    test_losses = []

    gradient_magnitudes = {layer : [] for layer in layers_to_plot }

    for ep in range(epochs):
        model.train()
        pred_correct = 0
        ep_loss = 0.
        for i, (inputs, labels) in enumerate(tqdm(trainloader)):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            pred_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            ep_loss += loss.item()
        for layer_idx in layers_to_plot:
            specific_layer_index = layer_idx*2
            layer_gard = model[specific_layer_index].weight.grad
            if layer_gard is not None:
                layer_gradients = (layer_gard.norm(2)**2).sum()
                gradient_magnitudes[layer_idx].append(layer_gradients)
        train_accs.append(pred_correct / len(trainset))
        train_losses.append(ep_loss / len(trainloader))
        # Record gradient magnitudes

        model.eval()
        with torch.no_grad():
            for loader, accs, losses in zip([valloader, testloader], [val_accs, test_accs], [val_losses, test_losses]):
                correct = 0
                total = 0
                ep_loss = 0.
                for inputs, labels in loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    ep_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                accs.append(correct / total)
                losses.append(ep_loss / len(loader))
        

        print('Epoch {:}, Train Acc: {:.3f}, Val Acc: {:.3f}, Test Acc: {:.3f}'.format(ep, train_accs[-1], val_accs[-1], test_accs[-1]))

    if layers_to_plot is not None:
        plot_average_gradient_magnitudes(gradient_magnitudes, layers_to_plot)

    return model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses

def Q_6_2_5():
    """
    Trains a neural network model with skip connections and plots its training, validation, and test accuracies.

    The model architecture includes skip connections between every 30 layers starting from layer 0.

    Layers 0, 30, 60, 90, 95, and 99 are tracked for their average gradient magnitudes during training.
    """
    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')

    output_dim = len(train_data['country'].unique())
    model_params = [nn.Linear(2, 4), nn.ReLU()] + [nn.Linear(4, 4), nn.ReLU()] * 98 + [nn.Linear(4, output_dim)]
    model = nn.Sequential(*model_params)

    layers_to_track = [0, 30, 60, 90, 95, 99]
    train_model_with_average_gradient_magnitudes(train_data, val_data, test_data, model=model, layers_to_plot=layers_to_track,layers_num=2)

def Q_6_2_6():
    """
    Trains a neural network model with skip connections and plots its training, validation, and test accuracies.

    The model architecture includes skip connections between every 30 layers starting from layer 0.

    Layers 0, 30, 60, 90, 95, and 99 are tracked for their average gradient magnitudes during training.
    """
    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')
    output_dim = len(train_data['country'].unique())
    model = SkipConnection(2,output_dim);
    layers_to_track = [0, 30, 60, 90, 95, 99]
    train_model_with_average_gradient_magnitudes(train_data, val_data, test_data, model=model, layers_to_plot=layers_to_track)


def procces_data(data):
    """
    Processes the input data by applying a sinusoidal transformation.

    Parameters:
        data (numpy.ndarray): Input data to be processed.

    Returns:
        torch.Tensor: Processed data as a PyTorch tensor.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    new_model_inp = np.zeros((data.shape[0], data.shape[1] * 10))
    alphas = np.arange(0.1, 1.05, 0.1)
    for i in range(data.shape[1]):
        for j, a in enumerate(alphas):
            new_model_inp[:, i * len(alphas) + j] = np.sin(a * data[:, i])
    return torch.tensor(new_model_inp, dtype=torch.float32, device=device)


def Q_6_2_1_7():
    """
    Trains neural network models with and without sinusoidal representation and plots their training, validation, and test accuracies.
    """
    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')
    output_dim = len(train_data['country'].unique())
    batch_size = 256
    lr = 0.001
    epochs = 50
    model = []
    
    model3_para = [nn.Linear(20, 16), nn.ReLU(),  # hidden layer 1
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 2
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 3
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 4
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 5
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 6
             nn.Linear(16, output_dim)  # output layer
             ]
    model_with_sin = nn.Sequential(*model3_para)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_with_sin.to(device)
    print('Using device:', device)
    trainset = torch.utils.data.TensorDataset(procces_data(torch.tensor(train_data[['long', 'lat']].values)).float(),
                                              torch.tensor(train_data['country'].values).long())
    valset = torch.utils.data.TensorDataset(procces_data(torch.tensor(val_data[['long', 'lat']].values)).float(),
                                            torch.tensor(val_data['country'].values).long())
    testset = torch.utils.data.TensorDataset(procces_data(torch.tensor(test_data[['long', 'lat']].values)).float(),
                                             torch.tensor(test_data['country'].values).long())

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1024, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model_with_sin.parameters(), lr=lr)
    train_accs = []
    val_accs = []
    test_accs = []
    train_losses = []
    val_losses = []
    test_losses = []
    for ep in range(epochs):
        model_with_sin.train()
        pred_correct = 0
        ep_loss = 0.
        for i, (inputs, labels) in enumerate(tqdm(trainloader)):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model_with_sin(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            pred_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            ep_loss += loss.item()
        train_accs.append(pred_correct / len(trainset))
        train_losses.append(ep_loss / len(trainloader))
        model_with_sin.eval()
        with torch.no_grad():
            for loader, accs, losses in zip([valloader, testloader], [val_accs, test_accs], [val_losses, test_losses]):
                correct = 0
                total = 0
                ep_loss = 0.
                for inputs, labels in loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model_with_sin(inputs)
                    loss = criterion(outputs, labels)
                    ep_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                accs.append(correct / total)
                losses.append(ep_loss / len(loader))

        print('Epoch {:}, Train Acc: {:.4f}, Val Acc: {:.4f}, Test Acc: {:.4f}'.format(ep, train_accs[-1], val_accs[-1],
                                                                                       test_accs[-1]))
    plt.figure()
    plt.plot(train_losses, label='Train', color='red')
    plt.plot(val_losses, label='Val', color='blue')
    plt.plot(test_losses, label='Test', color='green')
    plt.title('Losses for ith sinus representation')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(train_accs, label='Train', color='red')
    plt.plot(val_accs, label='Val', color='blue')
    plt.plot(test_accs, label='Test', color='green')
    plt.title('Accs for with sinus representation')
    plt.legend()
    plt.show()

    plot_decision_boundaries(model_with_sin, test_data[['long', 'lat']].values, test_data['country'].values, 'Decision Boundaries for implicit representation', implicit_repr=True)
    print('withsin: ', 'Train Accuracy:', train_accs[-1], 'Validation Accuracy:', val_accs[-1], 'Test Accuracy:',
          test_accs[-1])

    # train regualr model
    train_data = pd.read_csv('train.csv')
    model = [nn.Linear(2, 16), nn.ReLU(),  # hidden layer 1
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 2
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 3
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 4
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 5
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 6
             nn.Linear(16, output_dim)  # output layer
             ]
    model = nn.Sequential(*model)
    model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = train_model(train_data,
                                                                                                val_data, test_data,
                                                                                                model)
    plt.figure()
    plt.plot(train_losses, label='Train', color='red')
    plt.plot(val_losses, label='Val', color='blue')
    plt.plot(test_losses, label='Test', color='green')
    plt.title('Losses for regular representation')
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(train_accs, label='Train', color='red')
    plt.plot(val_accs, label='Val', color='blue')
    plt.plot(test_accs, label='Test', color='green')
    plt.title('Accs for without sinus representation')
    plt.legend()
    plt.show()

    plot_decision_boundaries(model, test_data[['long', 'lat']].values, test_data['country'].values, 'Decision Boundaries without sinus representation', implicit_repr=False)
    print('without Sin: ', 'Train Accuracy:', train_accs[-1], 'Validation Accuracy:', val_accs[-1], 'Test Accuracy:',
          test_accs[-1])



    
    
def plot_training_accuracy(train_accs, val_accs, test_accs, label):
    """
    Plots the training, validation, and test accuracies.

    Parameters:
        train_accs (list): List of training accuracies.
        val_accs (list): List of validation accuracies.
        test_accs (list): List of test accuracies.
        label (str): Label for the plot.
    """
    plt.plot(range(1, len(train_accs) + 1), train_accs, label=f'Training Accuracy ({label})', marker='o')
    plt.plot(range(1, len(val_accs) + 1), val_accs, label=f'Validation Accuracy ({label})', marker='o')
    plt.plot(range(1, len(test_accs) + 1), test_accs, label=f'Test Accuracy ({label})', marker='o')


    

    

















    

    
    


        


if __name__ == '__main__':
    # seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
