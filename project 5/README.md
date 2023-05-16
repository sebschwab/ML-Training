# Comparing Neural Nets

Sebastian Schwab

> In this project, I use several Neural Nets to advance the Lorentz equation and perform predictions. Since the Lorenz is so chaotic, only a well-trained Neural Network can effectively predict its next state.

## Sec. I. Introduction and Overview

Neural Networks can do more than classification. In this project I attempt to develop models to predict the future states of an equation. The equation features a high level of chaos, so it is a very difficult prediction. This equation, the Lorenz equation, has multiple different parameters so I test which ones generate the most chaos before testing several differnent Neural Nets. The ones I test are the FFNN, LSTM, RNN, and ESN.

## Sec. II. Theoretical Background

#### The Lorenz Equation

The Lorentz system is a system of differential equations that are commonly used in various fields of science, including physics, meteorology, and chaos theory. The system is noted for its chaotic solutions for certain parameters and initial conditions.

The system of equations is as follows:

dx/dt = σ(y-x)

dy/dt = x(ρ-z) - y

dz/dt = xy - βz

Here, x, y, and z make up the system state, t is time, and σ, ρ, and β are system parameters.

#### Neural Nets

A Neural Network can be trained to learn the underlying dynamics of this system, and then used to predict the evolution of the system over time. The input to the network would typically be the current state of the system (x, y, z) and possibly the current time t, and the output would be the rates of change (dx/dt, dy/dt, dz/dt), which can then be used to update the system state. This is essentially a form of time series prediction, with the twist that the "future" is not just dependent on the "past" but also on the current system state.

The learning process involves training the network on a dataset of system states and corresponding rates of change. The network learns to approximate the function that maps from the current system state to the rates of change. Once the network is trained, it can be used to predict the future states of the system from the current state, effectively "advancing" the system in time.

#### Types of Neural Nets

- A Feed-Forward Neural Network (FFNN) used for the Lorenz problem learns to map current states to their rates of change, but lacks memory of past states. 
- A Recurrent Neural Network (RNN) introduces memory by looping outputs back into inputs, allowing it to model temporal dependencies. 
- Long Short-Term Memory (LSTM), a special kind of RNN, uses gating mechanisms to control information flow, making it better at learning long-term dependencies. 
- An Echo State Network (ESN), another type of RNN, features a fixed, randomly initialized reservoir that projects input into a high-dimensional space, and only the output weights are learned, offering computational efficiency.



## Sec. III. Algorithm Implementation and Development 

#### Part 1: Hyperparameter Tuning -Rho

To begin, I wanted to see how well the FFNN performed on the Lorentz equation time series. First I defined the Lorenz equation in python and set the following hyperparameters:

```
def lorenz_deriv(x_y_z, t0, sigma=sigma, beta=beta, rho=rho):
    x, y, z = x_y_z
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
    
dt = 0.01
T = 8
t = np.arange(0,T+dt,dt)
beta = 8/3
sigma = 10
rho = 28
```
The hyperparameter I will be changing is Rho, and I will be testing it with a Feed-Forward network. The class and activation functions are below:

```
# Define activation functions
def logsig(x):
    return 1 / (1 + torch.exp(-x))

def radbas(x):
    return torch.exp(-torch.pow(x, 2))

def purelin(x):
    return x

# Define the model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(in_features=3, out_features=10)
        self.fc2 = nn.Linear(in_features=10, out_features=10)
        self.fc3 = nn.Linear(in_features=10, out_features=3)
        
    def forward(self, x):
        x = logsig(self.fc1(x))
        x = radbas(self.fc2(x))
        x = purelin(self.fc3(x))
        return x
```

I also used these optimizer and loss functions:

```
# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

Here is how I trained the model:

```
# Train the model
for epoch in range(30):
    optimizer.zero_grad()
    outputs = model(nn_input)
    loss = criterion(outputs, nn_output)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, loss={loss.item():.4f}")
```

#### Part 2: LSTM Testing

The next model I designed was an LSTM and its class looks like this:

```
# Define the model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=3, hidden_size=10, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=10, hidden_size=10, num_layers=1, batch_first=True)
        self.fc = nn.Linear(in_features=10, out_features=3)
        
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = logsig(x)
        x, _ = self.lstm2(x)
        x = radbas(x)
        x = self.fc(x)
        x = purelin(x)
        return x
```

I used the same optimizer and loss function:

```
# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

#### Part 3: RNN Testing

Next up was a Recursive Neural Network, and to create this I had to modify both the class definition and its loss function and optimizer:

```
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.rnn = nn.RNN(input_size=3, hidden_size=10, num_layers=1, batch_first=True)
        self.fc = nn.Linear(in_features=10, out_features=3)
        
    def forward(self, x):
        x, _ = self.rnn(x)
        x = logsig(x)
        x = self.fc(x)
        x = purelin(x)
        return x

# Create model instance and move to device
model = MyModel().to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

#### Part 4: Echo State Network Testing

This network was the most different from the other NNs, and it used a reservior system to amplify the RNN method. It was also the most difficult to implement.

```
class ESN1(nn.Module):
    def __init__(self, input_dim, reservoir_dim, output_dim):
        super(ESN1, self).__init__()
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.output_dim = output_dim

        self.Win = nn.Parameter(torch.randn(reservoir_dim, input_dim) / np.sqrt(input_dim), requires_grad=False)
        self.W = nn.Parameter(torch.randn(reservoir_dim, reservoir_dim) / np.sqrt(reservoir_dim), requires_grad=True)
        self.Wout = nn.Parameter(torch.zeros(output_dim, reservoir_dim), requires_grad=True)

        # Optional: initialize W with spectral radius < 1 for stability
#         spectral_radius = torch.max(torch.abs(torch.eig(self.W)[0]))
#         self.W.data = self.W.data / spectral_radius

    def forward(self, x):
        batch_size, sequence_length, _ = x.size()

        reservoir_state = torch.zeros(batch_size, self.reservoir_dim, device=x.device, dtype=x.dtype)
        for t in range(sequence_length):
            reservoir_state = torch.tanh(torch.mm(x[:, t, :], self.Win.T) + torch.mm(reservoir_state, self.W.T))
        
        out = torch.mm(reservoir_state, self.Wout.T)
        return out

# Usage
input_dim = nn_input.shape[1]
reservoir_dim = 100  # Define as per your need
output_dim = nn_output.shape[1]
```

I also had to use a different optimizer and loss function:

```
# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
```

## Sec. IV. Computational Results

#### Part 1: Hyperparameter Tuning -Rho

![image](https://github.com/sebschwab/ML-Training/assets/129328983/5685bff3-63f3-4e18-b197-39d5b3f1036f)

Figure 1: NN results with Rho = 10

![image](https://github.com/sebschwab/ML-Training/assets/129328983/144496fc-9290-4582-a75e-f199abe66ac4)

Figure 2: NN results with Rho = 17

![image](https://github.com/sebschwab/ML-Training/assets/129328983/dc793c0f-be99-4595-8b7f-b0135b6d35b0)

Figure 3: NN results with Rho = 28

![image](https://github.com/sebschwab/ML-Training/assets/129328983/b776c72c-3782-4344-9b6b-d77782b7eac5)

Figure 4: NN results with Rho = 35

![image](https://github.com/sebschwab/ML-Training/assets/129328983/478195da-c4a3-430e-9da7-bed002bfd77b)

Figure 5: NN with results Rho = 40

#### Part 2: LSTM Testing

![image](https://github.com/sebschwab/ML-Training/assets/129328983/9080981d-3e5b-4602-99d7-e1ed06cea176)

Figure 6: LSTM results with Rho = 28

#### Part 3: RNN Testing

![image](https://github.com/sebschwab/ML-Training/assets/129328983/bca0e579-ca77-48ed-b631-5328086a958a)

Figure 7: RNN results with Rho = 28

#### Part 4: Echo State Network Testing

![image](https://github.com/sebschwab/ML-Training/assets/129328983/8f157784-bba9-470e-98d6-912b36cfba4d)

Figure 8: ESN results with Rho = 28

## Sec. V. Summary and Conclusions

#### Part 1:

When testing the basic NN with different values for Rho - I found that 28 caused a huge spike in chaos. The loss factor is raised by over a multiple of 10. Since this dataset seemed the most chaotic, I chose to use this over the rest of the Neural Nets. The FFNN also performed very well in general for this prediction time series.

#### Part 2-4

From testing all of these Neural Nets I have been able to discover a few pieces of key information. The FFNN is very simple, yet for certain tasks it can perform incredibly well. This NN ran the fastest, and had the lowest loss after 30 epochs. This proved incredibly well for the Lorenz predictions, possibly because the system is so chaotic that having a memory only provides more distraction.

The LSTM performed ok, but not great. It was quick but not very accurate.

The RNN had a very high accuracy as well, although the run time took significantly longer than the FFNN. The computational power needed was much higher, so I would assume this should only be run on high end professional GPUs.

The ESN had the worst performance of all, I beleive this dataset was not well suited for the ESN and so it had a lot of difficulty.

