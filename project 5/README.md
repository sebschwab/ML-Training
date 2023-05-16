# Comparing Neural Nets

Sebastian Schwab

> In this project, I use several Neural Nets to advance the Lorentz equation and perform predictions. Since the Lorentz is so chaotic, only a well-trained Neural Network can effectively predict its next state.

## Sec. I. Introduction and Overview

Neural Networks can do more than classification. In this project I attempt to develop models to predict the future states of an equation. The equation features a high level of chaos, so it is a very difficult prediction. This equation, the Lorentz equation, has multiple different parameters so I test which ones generate the most chaos before testing several differnent Neural Nets. The ones I test are the FFNN, LSTM, RNN, and ESN.

## Sec. II. Theoretical Background

#### The Lorentz Equation

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

- A Feed-Forward Neural Network (FFNN) used for the Lorentz problem learns to map current states to their rates of change, but lacks memory of past states. 
- A Recurrent Neural Network (RNN) introduces memory by looping outputs back into inputs, allowing it to model temporal dependencies. 
- Long Short-Term Memory (LSTM), a special kind of RNN, uses gating mechanisms to control information flow, making it better at learning long-term dependencies. 
- An Echo State Network (ESN), another type of RNN, features a fixed, randomly initialized reservoir that projects input into a high-dimensional space, and only the output weights are learned, offering computational efficiency.



## Sec. III. Algorithm Implementation and Development 



```

```
## Sec. IV. Computational Results



## Sec. V. Summary and Conclusions


