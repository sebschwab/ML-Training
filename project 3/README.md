# Computing Correlation Using AI

Sebastian Schwab

> In this project, I use Python to interpret and analyze the MNIST dataset, which includes 70,000 grayscale images of handwritten digits 0-9. Each image is also
labeled with its correct digit. I use several different training machine and learning algorithms to compare their effectiveness at identifying and classifying these handwritten digits.

## Sec. I. Introduction and Overview

The MNIST data set was created in 1998 and consists of over 70,000 grayscale images of handwritten digits 0-9. The purpose of this dataset is to train and then subsequently test different learning algorithms and compare their effectiveness. While it is easy for a human to identify any written number, describing to a computer is difficult. So, algorithms such as the Singular Value Decomposition, Linear Classifier (LDA), Support Vector Machine, and decision tree are used to group the data together by similarities and try to create an accurate label.


## Sec. II. Theoretical Background

#### Singular Value Decomposition

One way to factorize data is known as the Singular Value Decomposition and is given by:

![image](https://user-images.githubusercontent.com/129328983/232873543-1fab45b1-e571-4af3-97d5-c56595448f93.png)

A visual representation of this is:

![image](https://user-images.githubusercontent.com/129328983/232873669-32779f1c-0be1-4e82-97a1-d0d0ba95b7e0.png)

#### Interpretation of SVD: (U, $\Sigma$, V)

U: The U vectors most similarly represent eigenvectors of the dataset. Although a more accurate name for them are "left singular vectors", what they truly represent are templates that capture a specific feature or pattern within the images. Every image can be reconstructed back together from addition of the U vectors.

$\Sigma$: The $\Sigma$ vector matrix contains the singular values. These values are within a diagonal matrix, and represent the importance of corresponding basis vectors U and V. If the singular value is higher, it represents a higher variance given by the U and V vectors and therefore a higher importance in the dataset.

V: The V matrix contains the singular vectors. They describe the combination process for reconstructing the images.

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
In order to use the data I had to reshape each image into its own column vector. I then performed an SVD of the data X and plotted the singular value spectrum (s):
```
# Reshape each image into a column vector and store them in a data matrix
X_reshaped = np.reshape(X, (X.shape[0], -1)).T

# Perform SVD analysis on the data matrix
U, s, Vt = np.linalg.svd(X_reshaped, full_matrices=False)
```
Finally, I created a 3D plot that displays three different V-modes and colors each digit separately. I chose to do columns 2, 3, and 5.
```
# Select columns 2, 3, and 5 from the Vt matrix
selected_modes = Vt[:,[1, 2, 4]]

# Project the data onto the selected modes
projected_data = np.dot(X_reshaped.T, selected_modes)

# Get the digit labels from the mnist dataset
labels = mnist.target.astype(int)
```

#### Part II: Linear Discriminant Analysis

In this section, my first goal was to design an LDA that could resonably identify between two different digits. I did this by creating a mask over the dataset that said which digits should and should not be looked at. Then I split that data into a training and test data set, and used the training to compare to test.
```
# Select two digits to classify
digit_1, digit_2 = 4, 9

# Filter the data to only include the selected digits
mask = np.logical_or(labels == digit_1, labels == digit_2)
X_filtered = X[mask]
labels_filtered = labels[mask]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_filtered, labels_filtered, test_size=0.3, random_state=42)

# Train an LDA classifier
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
```
Then I added one more digit for my comparison, doing this only required adding more data to my training and test set. So I simply updated the mask:
```
# Select three digits to classify
digit_1, digit_2, digit_3 = 9, 4, 7

# Filter the data to only include the selected digits
mask = np.logical_or.reduce((labels == digit_1, labels == digit_2, labels == digit_3))
X_filtered = X[mask]
labels_filtered = labels[mask]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_filtered, labels_filtered, test_size=0.3, random_state=42)

# Train an LDA classifier
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
```


## Sec. IV. Computational Results



## Sec. V. Summary and Conclusions


