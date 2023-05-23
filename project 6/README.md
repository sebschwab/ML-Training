# Evaluating a Shallow Recurrent Decoder

Sebastian Schwab

> In this project, I use a Shallow Recurrent Decoder to train and predict sea-surface temperatures across the globe. I then evaluate its performance by varying several different hyperparameters

## Sec. I. Introduction and Overview

This project uses a SHRED (Shallow Recurrent Decoder), which is a recurrent neural network (RNN) architecture designed for processing and predicting sequential data, particularly sensor measurements or time series data. It consists of an encoder-decoder structure where the encoder captures temporal dependencies and extracts features from the input sequences, while the decoder generates predictions. Despite its shallow architecture, SHRED is capable of effectively capturing temporal patterns and dependencies in the data. 

I use a SHRED to build a prediction series for all sea-surface temperatures across the globe. After being trained I assess its performance with a varying time lag, and number of sensors. I also test its ability to read through noise.


## Sec. II. Theoretical Background

#### SHRED's

SHRED (SHallow REcurrent Decoder) models are a network architecture that merges a recurrent layer (LSTM) with a shallow decoder network (SDN) to reconstruct high-dimensional spatio-temporal fields from a trajectory of sensor measurements of the field. More formally, the SHRED architecture can be written as

![image](https://github.com/sebschwab/ML-Training/assets/129328983/57b93ec0-dfa9-4dc6-a632-b024d5a9b7ce)


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


