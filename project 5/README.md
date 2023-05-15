# Comparing Neural Nets

Sebastian Schwab

> In this project, I use Python to interpret and analyze the MNIST dataset, which includes 70,000 grayscale images of handwritten digits 0-9. Each image is also
labeled with its correct digit. I use several different neural nets and learning algorithms to compare their effectiveness at identifying and classifying these handwritten digits.

## Sec. I. Introduction and Overview

The MNIST data set was created in 1998 and consists of over 70,000 grayscale images of handwritten digits 0-9. The purpose of this dataset is to train and then subsequently test different learning algorithms and compare their effectiveness. While it is easy for a human to identify any written number, describing to a computer is difficult. With this project, I attempt to create a basic neural net using the pytorch module and use it to classify these numbers. I also use algorithms such as the Singular Value Decomposition, Support Vector Machine, and decision tree are used to group the data together by similarities and try to create an accurate label.

## Sec. II. Theoretical Background

#### Neural Nets

The Neural Net I created takes an input, passes it through multiple layers where each layer transforms the data by applying mathematical operations, and then gives an output, which is a prediction. The model learns by adjusting these transformations based on how far its predictions are from the actual values, aiming to minimize this difference over multiple rounds of training.

In order to optimize a Neural Net, several hyperparameters are set. They are:

- Learning rate (lr): This is set to 0.01. The learning rate is a multiplier used during the updating of the model's weights. A smaller learning rate means the model learns slower, while a larger learning rate can speed up learning but might overshoot the optimal solution.

- Number of epochs (num_epochs): This is set to 1000. An epoch is a full pass through the entire training dataset. The more epochs, the more the model gets to learn from the data, but there's also a risk of overfitting if the number of epochs is too high.

- Size of the hidden layers: The first hidden layer has 64 neurons (fc1), and the second hidden layer has 32 neurons (fc2). These sizes determine the capacity of the model to learn complex patterns.

- Optimizer: The Stochastic Gradient Descent (SGD) optimizer is used. An optimizer is an algorithm that adjusts the weights of the model during training based on the gradient of the loss function.

- Loss function: The Mean Squared Error (MSE) loss function is used. The loss function measures the difference between the model's predictions and the actual values, guiding the optimizer on how to adjust the weights.

#### Feedforward Neural Network (FFNN) vs Long Short-Term Memory network (LSTM) 

In testing my neural test, I create two different models: a Feedforward Neural Network (FFNN) and Long Short-Term Memory network (LSTM). An FFNN is the simplest type of neural network; information moves in only one direction, from input to output. They do not have any memory of past inputs.

Unlike FFNNs, LSTMSs have loops in their network structure that allow information to be passed from one step in the sequence to the next. This gives FFNNs a kind of memory that can be used to process sequences of data, making them particularly useful for tasks like time-series prediction or natural language processing. 

#### SVM and Decision Trees



## Sec. III. Algorithm Implementation and Development 



```

```
## Sec. IV. Computational Results



## Sec. V. Summary and Conclusions


