# Machine Learning Training

> This repository contains a code series using Python to train a deep neural net.

[project 1](https://github.com/sebschwab/ML-Training/tree/main/project%201)

## Part One

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

## Part Two

Next, I decided to make a 2D loss landscape by fixing two parameters and sweeping through a large range of values for the other two.

![image](https://user-images.githubusercontent.com/129328983/231048385-0b2635eb-925a-44a7-868c-3c49b2c4fbd7.png)

I created a range of values for each parameter that, when swept through, would include the optimal values calculated from part 1. I also calculated the minimum loss from each of these combinations. Below are the graphs of these sweeps:

![image](https://user-images.githubusercontent.com/129328983/231048634-36f8374e-40ed-4270-bf2b-6453e59daffb.png)

![image](https://user-images.githubusercontent.com/129328983/231048648-625712c4-2163-4dee-b1a2-c63a50358ee7.png)

![image](https://user-images.githubusercontent.com/129328983/231048710-0a719909-50e6-4a17-8369-354de653520c.png)

![image](https://user-images.githubusercontent.com/129328983/231048741-941138b1-3e0e-4107-b042-4be16bdb2334.png)

![image](https://user-images.githubusercontent.com/129328983/231048776-61ea3ab9-8f71-4880-93d1-599996160e61.png)

This data shows me where local minimas are, and I can determine that some combinations of variables control a larger amount than others. This implies they have a higher control over the output of the model.

## Part Three

For the next part of my project, I decided to model different kinds of functions for the data set. I tested a linear, parabolic, and 19th degree polynomial. To begin, I split up my data into two parts: test and train.

The train data would be the first 20 values, which I would use to test on the final 10 values.

![image](https://user-images.githubusercontent.com/129328983/231049468-e1d08ca3-c147-4ca7-a5a2-210ecc00d3e8.png)

To create each of my models, I used the numpy polyval and polyfit functions. First I would create a function with optimal coefficients with the training data, then use that function to fit onto the test data. Here is the code for each three functions I created:

![image](https://user-images.githubusercontent.com/129328983/231049702-e5c5289e-b2a4-487c-8f61-57d958869205.png)

Next I printed out my results:

![image](https://user-images.githubusercontent.com/129328983/231049787-87b54c4a-de0e-4855-b58d-8b2edc2af3c2.png)

![image](https://user-images.githubusercontent.com/129328983/231049836-8a57619f-1709-4fc0-a9d9-9ce6172c6766.png)

![image](https://user-images.githubusercontent.com/129328983/231049879-49f48d1d-99b0-452f-8e39-25f01f37c6f9.png)

From these results, I was able to determine the best fit for each data set. The parabola fit the training data the best, which makes sense since the data most closely resembles that function shape. However, the line model fit the test data the best. Another conclusion I drew is that the 19th degree polynomial is certainly incorrect. The function overfit the data and could not be used on the test data at all.

## Part Four

Finally, I decided to test the data in another way. This time I used the first and last 10 values as the training set, and the middle 10 as the test values. I performed the same function modeling, linear, parabolic, and 19th degree polynomial. These were the results I acheived:

![image](https://user-images.githubusercontent.com/129328983/231051690-1231526b-e602-48a1-b1b6-e18e6519b177.png)

![image](https://user-images.githubusercontent.com/129328983/231050478-1cfb901c-a604-4fc3-962a-7ed483f20c3c.png)

This data shows a very similar pattern to the other training set. This leads me to believe that certianly 19th degree polynomials are too high and very prone to overfitting. It was interesting having a large gap in my data set because it caused the parabolic and linear functions to be very similar. They even overlap each other in the first graph. Their errors are very similar as well, therefore I conclude that either of those functions are appropriate to model my data.
