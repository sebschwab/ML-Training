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
I wanted to visualize the find the ideal rank (r) for the digit space, so to calculate that I performed the following:
```
total_energy = np.sum(s**2)
cumulative_energy = np.cumsum(s**2)
threshold = 0.95 * total_energy  # You can adjust the threshold as needed

# Find the optimal rank
optimal_rank = np.argmax(cumulative_energy > threshold) + 1

print(f"Optimal rank (r) for image reconstruction: {optimal_rank}")
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

#### Part III: Support Vector Machine

Next I wanted to test the data categorization power of an SVM. To do so, I first tested the accuracy of the learning algorithm with no masks. This means I used all 10 digits as comparison from one another. In performing this test, I wanted to compare against the training set but the data was far too large. The computation power needed to analyze such a large dataset was not in my ability. However I was able to fully test the test dataset:
```
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train an SVM classifier
svm = SVC(kernel='rbf', C=5, gamma='scale', random_state=42)
svm.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = svm.predict(X_test)
```
After doing all the digit values, I wanted to be able to compare the algorithm's ability to detect between two separate digits from an LDA's. So I rewrote my function with a mask for only two digits:
```
# Select two digits to classify
digit_1, digit_2 = 9, 4

# Filter the data to only include the selected digits
mask = np.logical_or(labels == digit_1, labels == digit_2)
X_filtered = X[mask]
labels_filtered = labels[mask]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_filtered, labels_filtered, test_size=0.3, random_state=42)

# Train an SVM classifier
svm = SVC(kernel='rbf', C=5, gamma='scale', random_state=42)
svm.fit(X_train, y_train)
```

#### Part IV: Decision Tree

Finally I wanted to compare these supervised machine learning algorithms with another, the decision tree. I followed the same process as I did for the SVM and started by comparing all 10 digits to each other:
```
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train a Decision Tree classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = dt.predict(X_test)
```
Then I adapted my code to include a mask that only compares two digits at a time:
```
# Select two digits to classify
digit_1, digit_2 = 4, 9

# Filter the data to only include the selected digits
mask = np.logical_or(labels == digit_1, labels == digit_2)
X_filtered = X[mask]
labels_filtered = labels[mask]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_filtered, labels_filtered, test_size=0.3, random_state=42)

# Train a Decision Tree classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
```

## Sec. IV. Computational Results

### How To interpret Results:

In parts II-IV the results are in the following form: accuracy -> confusion matrix -> classification report.

To analyze these results, first note the accuracy is simply a percentage of success achieved. It is the simplest and most direct metric.

The confusion matrix shows the number of successful identifications along the diagonal (notice the larger numbers in that pattern). This shows the number of successes broken down by each digit.

Finally, the classification report shows many different accuracy ratings. Precision and recall are very similar and essentially mean how succesfull was each test. The report is broken down by digit and summed at the bottom.

#### Part I: SVD Analysis

![image](https://user-images.githubusercontent.com/129328983/234186375-bffc9717-3194-4605-a52d-a0a59be8b505.png)

Figure 1: Singular Value Spectrum of SVD

![image](https://user-images.githubusercontent.com/129328983/234186675-33115d97-61ff-440d-841c-fd3895a86946.png)

Figure 2: Optimal rank (r) of digit space

![image](https://user-images.githubusercontent.com/129328983/234186791-36d109a4-1a95-4f56-8233-e9f8c54dc4c9.png)

Figure 3: Three V-Modes labelled and colored in a 3D plot


#### Part II: Linear Discriminant Analysis

![image](https://user-images.githubusercontent.com/129328983/234187638-10bb7ac6-6e14-4275-9a9f-e301bf9b28f8.png)

Figure 4: The training data vs the test data for two digits broken down in an LDA. (worst case)

![image](https://user-images.githubusercontent.com/129328983/234187872-f5167c0b-fc63-4b81-8f44-fa8e649abfe6.png)

Figure 5: The training data vs the test data for two digits broken down in an LDA. (best case)

![image](https://user-images.githubusercontent.com/129328983/234187981-7da5d31b-244e-4559-ab58-a74ff2c9bc5a.png)
![image](https://user-images.githubusercontent.com/129328983/234188006-fd3c3054-e4ff-4e33-aefa-c1d8495d2754.png)

Figure 6: The training data vs the test data for three digits broken down in an LDA. (worst case)

![image](https://user-images.githubusercontent.com/129328983/234188215-3f516d8e-e27c-460a-a25c-447abbc7d96e.png)
![image](https://user-images.githubusercontent.com/129328983/234188239-14b7066d-c146-4e57-9b87-30ff4bfadacb.png)

Figure 7: The training data vs the test data for three digits broken down in an LDA. (best case)

#### Part III: Support Vector Machine

![image](https://user-images.githubusercontent.com/129328983/234188385-cdb501ea-efcc-4b5a-b703-48db669c1764.png)

Figure 8: The support vector machine classification for all 10 digits

![image](https://user-images.githubusercontent.com/129328983/234188486-f4fbc0eb-3817-478f-b65a-4125aef606af.png)

Figure 9: The SVM classification for only two digits (worst case)

![image](https://user-images.githubusercontent.com/129328983/234188651-b90c7308-e6af-4d9e-a837-356aa9249b14.png)

Figure 10: The SVM classification for only two digits (best case)

#### Part IV: Decision Tree

![image](https://user-images.githubusercontent.com/129328983/234189013-98ab83fa-36a0-468c-baef-4ec1c99d1d91.png)

Figure 10: The decision tree classification for all 10 digits

![image](https://user-images.githubusercontent.com/129328983/234189236-311754e2-0291-49a8-ad4f-fced9b655efa.png)

Figure 11: The decision tree classification for only two digits (worst case)

![image](https://user-images.githubusercontent.com/129328983/234189385-d6e3746b-47c3-44e8-b81c-f66ae8abfcf5.png)

Figure 12: The decision tree classification for only two digits (best case)

## Sec. V. Summary and Conclusions


