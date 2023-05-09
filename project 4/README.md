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

A Support Vector Macine is a classification algorithm that finds the optimal hyperplane to separate different features in a feature space. It is very adaptable and very powerful.

A Decision Tree acts like a flow chart to categorize data. Each node represents a decision used to split the data, and each leaf node is a category of that decision. The tree is built by selecting the best feature to split the data at each step.

## Sec. III. Algorithm Implementation and Development 

#### Part I: Feedforward Neural Net

The first thing I wanted to test was a basic dataset:
```
X = np.arange(0, 31)
Y = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41,
              40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])
```
I created a three-layer FFNN that used Schocastic Gradient Descent and a Least-Squared Error loss function by defining two things. The initialization and its "step direction". 

```
# Define model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```
Then I produced the results by creating a training set (20 values) and a test set (10 values). I tried two different way of splitting the data and compared the two resulting Least-Squared Errors.

#### Part II: MNIST Dataset - PCA

Next I wanted to test a larger data set. So, I loaded in the MNIST digit values and split the values into test and train sets. I also created loaders using the PyTorch module and appropriately reshaped the data:
```
# Load the MNIST dataset and apply transformations
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

# Create data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Combine train and test datasets
mnist_data = torch.cat([train_dataset.data, test_dataset.data], dim=0)
mnist_labels = torch.cat([train_dataset.targets, test_dataset.targets], dim=0)

# Flatten the images
mnist_data = mnist_data.view(mnist_data.size(0), -1).float() / 255.0
```

Next, I performed a PCA to reduce the dimensionality to 20. This is done by finding the 20 most significant directions (or modes) in the data that capture the most variation, which simplifies the data while retaining key information.

```
pca = PCA(n_components=20)
pca.fit(mnist_data)

# Get the first 20 PCA modes
pca_modes = pca.components_
```

#### Part III: MNIST Dataset - FFNN

I then wanted to use the FFNN I had created on this dataset, so I slightly modified the defining functions and kept the hyperparameters the same. The loss function was changed to Cross-Entropy Loss:

```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784) # flatten the input image
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
 ```
 
#### Part IV: MNIST Dataset - LSTM

I then moved on the test the LSTM neural net agains the MNIST dataset. It starts with a zeroed initial state, then reads through the input sequence, updating its state based on both the current input and its memory of past inputs. The final state is passed through a linear layer to produce the model's prediction for the input sequence:

```
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
 ```
 I changed the hyperparameters to the following and kept all else the same:
 ```
 # Hyperparameters
input_size = 28
sequence_length = 28
hidden_size = 128
num_layers = 2
num_classes = 10
num_epochs = 10
learning_rate = 0.01
```

#### Part V: MNIST Dataset - Support Vector Machine

Next, I wanted to test these results against a Support Vector Machine's classifcation. To create one, I split up the MNIST data into training and test sets and used the SkLearn package to fit the data:

```
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train an SVM classifier
svm = SVC(kernel='rbf', C=5, gamma='scale', random_state=42)
svm.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = svm.predict(X_test)
```
#### Part VI: MNIST Dataset - Decision Tree

I performed a very similar method to test the decision tree:

```
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train a Decision Tree classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = dt.predict(X_test)
```
## Sec. IV. Computational Results

#### Part I

![image](https://user-images.githubusercontent.com/129328983/236963266-a82438ac-0856-489e-a42e-9d4114481a5c.png)

Figure 1: LSE results for FFNN on array dataset, using first 20 as training.

![image](https://user-images.githubusercontent.com/129328983/236964033-db83a4a9-b088-4cd5-a3cf-b398b55f43a5.png)

Figure 2: Polynomial Fits from project 1 using same dataset (testing and training)

![image](https://user-images.githubusercontent.com/129328983/236963327-53dfe48c-81df-42ff-80d0-2eadecc6ccb9.png)

Figure 3: LSE results for FFNN on array dataset, using first and last 10 as training.

![image](https://user-images.githubusercontent.com/129328983/236964147-f53b80ba-9e74-4c17-832c-9108de81d1b9.png)

Figure 4: Polynomial Fits from project 1 using same dataset (testing and training)

#### Part II

![image](https://user-images.githubusercontent.com/129328983/236963390-76fa7fd8-ed97-41d8-86f5-a7cea945984f.png)

Figure 5: Digit representation of the first 20 PCA modes of the MNIST dataset

#### Part III

![image](https://user-images.githubusercontent.com/129328983/236963510-bd99b706-f8af-410e-a6ed-71e4eebfe0de.png)
![image](https://user-images.githubusercontent.com/129328983/236961618-f2140eba-60cb-42a2-bfee-b3fd80754a80.png)

Figure 6: Computational results and accuracy of the FFNN on the MNIST dataset

#### Part IV

![image](https://user-images.githubusercontent.com/129328983/236964325-db12ccd1-d833-4f68-aa7f-96e5d6dfe187.png)

Figure 7: Computational results and accuracy of the LSTM on the MNIST dataset

#### Part V

![image](https://user-images.githubusercontent.com/129328983/236963636-56d8a50e-5ae9-49fe-9625-69c257f1a35a.png)

Figure 8: Accuracy Score for the SVM classification

#### Part VI

![image](https://user-images.githubusercontent.com/129328983/236963705-024bb01e-65e6-4741-8411-297bd6a87cf9.png)

Figure 9: Accuracy Score for the Decision Treeclassification

## Sec. V. Summary and Conclusions

#### Part I

When comparing the Neural Net accuracies between the two different test sets, it appears that the test set that included a discontinous set of values (first and last ten numbers) performed marginally better. While both models had a significantly low LSE, the split training set perfomed better on both testing and training evaluations. Although both models had a higher test error than training error, the test error on the split set was below half that of the first set. I believe this is because a split data set has a more varied series of values and so prepares the FFNN better.

From these results I can clearly detect a much higher accuracy in the Neural Nets than in any kind of polynomial fit. Both Neural Nets had a < 0.01 error. While the polynomial fits ranged from an error of 6.0 - 200,000. It can definitely be reasoned that Neural Nets can fit data better, however this does come with a slightly higher computation cost.

#### Part II-VI

On all Neural Nets (no matter FFNN or LSTM) the first Epoch had the highest loss, and it continually decreased after each iteration. This can be seen in figure 6. Overall when comparing the FFNN and LSTM, both had similar accuracy. Although, the LSTM scored 1% higher than the FFNN. However, this came with an incredibly higher computation cost: the LSTM took over 10 minutes to compute on a laptop while the FFNN took about 2.

The best performing classification model was the Support Vector Machine. This one took an average amount of time (2 minutes) yet performed the best with an accuracy > 98%. The model does not even use a Neural Net, which makes me think its application is better suited for other tasks. Second and third place were the FFNN and LSTM, with a very high accuracy of 93% and 94%. Yet, these took by far the longest to compute. Finally, the fastest yet also most inaccurate was the Decision Tree with an accuracy of 86%.
