# Evaluating 

Sebastian Schwab

> In this project, I use a Shallow Recurrent Decoder to train and predict sea-surface temperatures across the globe. I then evaluate its performance by varying several different hyperparameters

## Sec. I. Introduction and Overview



## Sec. II. Theoretical Background

#### 

#### Neural Nets

A Neural Network can be trained to learn the underlying dynamics of this system, and then used to predict the evolution of the system over time. The input to the network would typically be the current state of the system (x, y, z) and possibly the current time t, and the output would be the rates of change (dx/dt, dy/dt, dz/dt), which can then be used to update the system state. This is essentially a form of time series prediction, with the twist that the "future" is not just dependent on the "past" but also on the current system state.

The learning process involves training the network on a dataset of system states and corresponding rates of change. The network learns to approximate the function that maps from the current system state to the rates of change. Once the network is trained, it can be used to predict the future states of the system from the current state, effectively "advancing" the system in time.

#### Types of Neural Nets

- A Feed-Forward Neural Network (FFNN) used for the Lorenz problem learns to map current states to their rates of change, but lacks memory of past states. 
- A Recurrent Neural Network (RNN) introduces memory by looping outputs back into inputs, allowing it to model temporal dependencies. 
- Long Short-Term Memory (LSTM), a special kind of RNN, uses gating mechanisms to control information flow, making it better at learning long-term dependencies. 
- An Echo State Network (ESN), another type of RNN, features a fixed, randomly initialized reservoir that projects input into a high-dimensional space, and only the output weights are learned, offering computational efficiency.



## Sec. III. Algorithm Implementation and Development 



## Sec. IV. Computational Results



## Sec. V. Summary and Conclusions


