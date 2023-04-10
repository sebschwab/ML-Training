# Machine Learning Training

> This repository contains a code series using Python to train a deep neural net.

I began with simply an array of 31 data points:

X=np.arange(0,31)

Y=np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41,
40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])

To train a function to best fit this data, first a type of function needs to be set. I started with a first-order trigometric function:

F(x) = Acos(Bx) + Cx + D

This function has three parameters that need to be optimized. To do this I used the scipy module "optimize". This optimize function requires a model to follow. So I used the Root Mean Squared Error method. This function is of the form:

E = sqrt((1/n)sum(f_xi-y)^2)

Where n is the size of the data set.

![image](https://user-images.githubusercontent.com/129328983/230983491-4ff4a814-fec0-4022-9500-6b6b7f8b085d.png)

