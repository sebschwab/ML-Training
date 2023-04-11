# Machine Learning Training

> This repository contains a code series using Python to train a deep neural net.

### Part One

I began with simply an array of 31 data points:

X=np.arange(0,31)

Y=np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41,
40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])

To train a function to best fit this data, first a type of function needs to be set. I started with a first-order trigometric function:

F(x) = Acos(Bx) + Cx + D

This function has three parameters that need to be optimized. To do this I used the scipy module "optimize". This optimize function requires a model to follow. So I used the Root Mean Squared Error method. This function is of the form:

E = (1/n)*sqrt(sum(f_xi-y)^2)

Where n is the size of the data set.

![image](https://user-images.githubusercontent.com/129328983/231047861-bff01f37-4a44-4d6b-a17b-e96b47afb60b.png)

To begin, I defined my data a set and objective function. I also set an initial guess for the AI to start from in its model.

![image](https://user-images.githubusercontent.com/129328983/231048007-46ae7c84-41d0-418b-8dcb-1e5e19de9380.png)

Then I optimized the data and printed the resulting optimal parameters and minimum error. The resulting graph is below:

![image](https://user-images.githubusercontent.com/129328983/231048105-735f2a55-49c7-449b-82cf-536bd25ddafc.png)

### Part 2

Next, I decided to make a 2D loss landscape by fixing two parameters and sweeping through a large range of values for the other two.

![image](https://user-images.githubusercontent.com/129328983/231048385-0b2635eb-925a-44a7-868c-3c49b2c4fbd7.png)

I created a range of values for each parameter that, when swept through, would include the optimal values calculated from part 1. I also calculated the minimum loss from each of these combinations. Below are the graphs of these sweeps:

![image](https://user-images.githubusercontent.com/129328983/231048634-36f8374e-40ed-4270-bf2b-6453e59daffb.png)

![image](https://user-images.githubusercontent.com/129328983/231048648-625712c4-2163-4dee-b1a2-c63a50358ee7.png)

![image](https://user-images.githubusercontent.com/129328983/231048710-0a719909-50e6-4a17-8369-354de653520c.png)

![image](https://user-images.githubusercontent.com/129328983/231048741-941138b1-3e0e-4107-b042-4be16bdb2334.png)

![image](https://user-images.githubusercontent.com/129328983/231048776-61ea3ab9-8f71-4880-93d1-599996160e61.png)


### Part Three
