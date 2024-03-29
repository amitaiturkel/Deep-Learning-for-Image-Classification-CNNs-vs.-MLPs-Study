
## Getting Started

To get started with the project, follow these steps:

### Download the Project

Clone this repository to your local machine by running the following command in your terminal:

`git clone https://github.com/amitaiturkel/Deep-Learning-for-Image-Classification-CNNs-vs.-MLPs-Study.git`
go to Deep-Learning-for-Image-Classification-CNNs-vs.-MLPs-Study folder 

### Set Up the Virtual Environment

Activate a virtual environment to isolate the project's dependencies. If you're using `virtualenv`, you can create and activate the environment with the following commands:

`python3 -m venv .venv` 
or run `python -m venv .venv`
and then
`source .venv/bin/activate`


### Install Dependencies

Use `poetry` to install the project's dependencies:

`poetry install`

### GUI BUGS

If you encounter issues with the graphical user interface (GUI) while running the project through a virtual machine (VM), particularly related to `matplotlib.pyplot`, please note that the VM environment may have limitations or configurations that affect GUI rendering.

To resolve this issue, it's recommended to run the script directly on your local machine instead of through a VM. This should help ensure proper functionality and display of the GUI components.

To install the required dependencies witout poetry, use the following command:

`pip install matplotlib torch scikit-learn pandas numpy` 


## 6 Multi-Layer Perceptrons

### 6.1 Optimization of an MLP

In this section, we delve into the optimization of **Multi-Layer Perceptrons (MLPs)**, gradually tweaking essential parameters to comprehend their significance in training neural networks. the code is in the file **NN_turtorial.py**

#### 6.1.1 Learning Rate

We will explore the impact of different **learning rates** (1., 0.01, 0.001, 0.00001) on training by plotting the validation loss of each epoch.

#### 6.1.2 Epochs

By training the network for **100 epochs** and plotting the loss over specific epochs (1, 5, 10, 20, 50, 100), we'll discern how the number of epochs influences the training process.

#### 6.1.3 Batch Norm

Adding a **batch normalization layer** (nn.BatchNorm1d) after each hidden layer, we'll compare the results with the regular model to understand its contribution to training.

#### 6.1.4 Batch Size

We'll investigate the effect of **batch size** (1, 16, 128, 1024) on training by adjusting the number of epochs accordingly. We'll analyze its impact on accuracy, speed, and stability.

### 6.2 Evaluating MLPs Performance

In this section, we'll train multiple MLPs and scrutinize their algorithmic choices to gain insights into their performance.

#### 6.2.1 Model Analysis

We'll address various questions to comprehensively analyze the models:

1. **Best Model Analysis:** We'll choose the model with the best validation accuracy, plot its training, validation, and test losses, and visualize its prediction space to assess generalization.

2. **Worst Model Analysis:** Similarly, we'll analyze the model with the worst validation accuracy to determine if it over-fitted or under-fitted the dataset.

3. **Depth of Network:** Using MLPs of width 16, we'll analyze the impact of the number of hidden layers on the training, validation, and test accuracy and discuss the advantages and disadvantages of increasing the number of layers.

4. **Width of Network:** Using MLPs of depth 6, we'll explore how the number of neurons in each hidden layer affects the training, validation, and test accuracy.

5. **Gradient Monitoring:** We'll monitor the magnitude of gradients for each layer during training to detect any vanishing or exploding gradients issues and propose potential solutions.

6. **Bonus Modification:**  we'll implement suggested modifications to address any observed gradient issues and evaluate their effectiveness.

7. **Implicit Representation:**  we'll implement implicit representation pre-processing to the data and compare its performance with a standard architecture trained on the data.



the following code is in **cnn.py**

### 7 Convolutional Neural Networks

In this section, we delve into Convolutional Neural Networks (CNNs), a potent class of deep learning models tailor-made for image-related tasks. Our focus is to apply CNNs to a practical real-world scenario: detecting deepfake-generated images. We'll work with a labeled dataset containing both authentic human faces and deepfake-generated ones.

#### 7.1 Dataset Overview

Our dataset consists of labeled images, categorized into real human faces and deepfake-generated ones. It's divided into training, validation, and test sets, each comprising both real and fake images.

#### 7.2 Task Overview

The task involves evaluating various baselines:

1. **XGBoost:** Utilize the XGBoost classifier with default settings.
2. **Training from Scratch:** Train a ResNet18 classifier on the training data, implementing logistic regression.
3. **Linear Probing:** Train a frozen ResNet18 model pretrained on ImageNet, optimizing only the final linear layer.
4. **Linear Probing with sklearn:** Extract feature representations using a pre-trained ResNet18 model and train a Logistic Regression model using sklearn.
5. **Fine-tuning:** Fine-tune a pre-trained ResNet18 model for binary classification.

#### 7.3 Experimental Setup

We'll experiment with the Adam optimizer and vary the learning rates within the range [0.00001, 0.1]. Training will be conducted with a batch size of 32 samples for 1 epoch, employing binary cross-entropy loss.

#### 7.4 Model Implementation

We provide skeleton code for CNN models and evaluation protocols in cnn.py, and for the XGBoost classifier in xg.py, to facilitate your experiments. This code serves as a starting point, enabling you to focus on training and evaluation.

#### 7.5 Key Questions

1. **Model Evaluation:** Identify the top-performing and worst-performing models based on test accuracy. Discuss the choice of learning rates for PyTorch CNN baselines and explain the observed trends.
2. **Sample Analysis:** Visualize 5 samples correctly classified by the best baseline but misclassified by the worst-performing baselines to gain deeper insights into model performance.


