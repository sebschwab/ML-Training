# Testing Machine Learning Algorithms

Sebastian Schwab

> In this project, I use Python to interpret and analyze the MNIST dataset, which includes 70,000 grayscale images of handwritten digits 0-9. Each image is also
labeled with its correct digit. I use several different neural nets and learning algorithms to compare their effectiveness at identifying and classifying these handwritten digits.

## Sec. I. Introduction and Overview

The MNIST data set was created in 1998 and consists of over 70,000 grayscale images of handwritten digits 0-9. The purpose of this dataset is to train and then subsequently test different learning algorithms and compare their effectiveness. While it is easy for a human to identify any written number, describing to a computer is difficult. So, algorithms such as the Singular Value Decomposition, Support Vector Machine, and decision tree are used to group the data together by similarities and try to create an accurate label.


## Sec. II. Theoretical Background



#### SVM and decision trees

A Support Vector Macine is a classification algorithm that finds the optimal hyperplane to separate different features in a feature space. It is very adaptable and very powerful.

A Decision Tree acts like a flow chart to categorize data. Each node represents a decision used to split the data, and each leaf node is a category of that decision. The tree is built by selecting the best feature to split the data at each step.

## Sec. III. Algorithm Implementation and Development 

#### Part I: SVD analysis

The first thing I did was load in the MNIST dataset:
```
mnist = fetch_openml('mnist_784')
X = mnist.data / 255.0  # Scale the data to [0, 1]
```


## Sec. IV. Computational Results





## Sec. V. Summary and Conclusions




