# Computing Correlation Using AI

Sebastian Schwab

> In this project, I use Python to interpret and analyze the "yalefaces.mat" file which includes over 2000 images of faces from different lightings.
> This data set was created in the early days of AI, and was one of the first facial recognition algorithms created.

## Sec. I. Introduction and Overview
Throughout this project I use a matrix to represent all of the photos together, this matrix X uses a compressed form of manipulation that turns every 2D image into a single column. Therefore, every column represents a single image. I use correlational matrices to compare which values are most highly related, and unrelated. I also try to find an eigenvector and SVD representation of the correlational matrix. Then compare the two methods to determine their accuracy.

## Sec. II. Theoretical Background
Correlational matrices are used to represent the data, to perform one you must dot product one matrix to the transpose of itself ($C = A*A^T$).

An eigenvector is a vector that when multiplied by a matrix A, produces a scalar multiple of A. 

Such as: $A * V = \delta * V$ where V is the eigenvector.

Another way to factorize data is known as the Singular Value Decomposition and is given by:

![image](https://user-images.githubusercontent.com/129328983/232873543-1fab45b1-e571-4af3-97d5-c56595448f93.png)

A visual representation of this is:

![image](https://user-images.githubusercontent.com/129328983/232873669-32779f1c-0be1-4e82-97a1-d0d0ba95b7e0.png)

It is my goal to use these manipulation methods to analyze and produce a meaningful way of identifying faces from this dataset.

## Sec. III. Algorithm Implementation and Development 

#### Part I

To begin, I loaded in the faces matrix and assigned them to X. Then I created a correlational matrix of the first 100 photos and plotted using the pcolor attribute in matplot.

```
results=loadmat('yalefaces.mat')
X=results['X']

# Extract the first 100 images from the matrix X
X_100 = X[:, :100]

# Compute the correlation matrix
C = np.dot(X_100.T, X_100)
```

#### Part II

Using the correlational matrix C, I determined to find the two most correlated and uncorrelated photos. To do this, I used the argmax and argmin methods to find the highest and lowest values on C. However, initially these values both fell along the diagonal which means the photos were most highly correlelated with themselves. To fix this I set all the values on the diagonal to be zero, and then a very high number:
```
# set diagonal elements to zero
np.fill_diagonal(C, 0)
# find the indices of the most highly correlated
max_idx = np.unravel_index(np.argmax(C), C.shape)

# set diagonal elements to 100000
np.fill_diagonal(C, 100000)
# find the indices of the most highly uncorrelated
min_idx = np.unravel_index(np.argmin(C), C.shape)
```

#### Part III
Next, I wanted to create smaller correlational matrices of only a 10x10 size to better understand the data. I chose random points along the 1000 photos to capture 10-photo segments and compare them. The random values can be seen in the array values:
```
values = [1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005]

fig, axs = plt.subplots(2, 5, figsize=(12, 5))
k = 0
for i in range(2):
    for j in range(5):
        # Extract the first 100 images from the matrix X
        X_10 = X[:, values[k]-1:values[k]+9]

        # Compute the correlation matrix
        C = np.dot(X_10.T, X_10)
        
        value = i *j
        axs[i, j].pcolor(C)
        axs[i, j].set_title(f'Matrix from {values[k]} to {values[k]}')
        k += 1
```

#### Part IV
In my next task I set out to create eigenvectors of the data. First, I created a correlational matrix of the entire dataset X. Then after creating the vectors, I sort the values from highest to lowest and plot the first six.
```
# Create the matrix Y = XXT
Y = np.dot(X, X.T)

# Find the eigenvectors and eigenvalues of Y
eigenvalues, eigenvectors = np.linalg.eig(Y)

# Sort the eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# Select the first six eigenvectors with the largest magnitude eigenvalue
U6 = sorted_eigenvectors[:, :6]
```

#### Part V
Next I SVD the matrix X and find the first six principle components. This was very straighforward in python thanks to numpy's comprehensive list of functions.
```
U_svd, S, Vt = np.linalg.svd(X, full_matrices=False)

# Find the first six principal component directions
PC = Vt[:6, :]
```

#### Part VI
After collecting the SVD Principal Components and the fist 6 eigenvectors, I compare the first of each to one another using the norm of differnce of their absolute values:
```
# Compare the first eigenvector v1 with the first SVD mode u1
u1 = U_svd[:, 0]
v1 = U6[:, 0]
diff_norm = np.linalg.norm(np.abs(v1) - np.abs(u1))
```

#### Part VII
Finally, I wanted to compute the accuracy of the SVD modes by comparing their variances. So I captured the total variance of the entire matrix, and then compared that to the resulting variance from each of the six SVD modes:

```
# Compute the SVD
U, S, Vt = np.linalg.svd(X)

# Compute the total variance
total_var = np.sum(X**2)

# Compute the variance captured by each mode
var_captured = (S**2) / total_var

# Compute the percentage of variance captured by each mode
percent_var_captured = var_captured * 100
```

## Sec. IV. Computational Results

#### Part I

![image](https://user-images.githubusercontent.com/129328983/232882077-826fd3ef-6e1f-40a1-86bd-6be0a5a14e00.png)

#### Part II

![image](https://user-images.githubusercontent.com/129328983/232882144-bbc0695c-9a50-49ac-8cd3-94e13e222f3e.png)

#### Part III

![image](https://user-images.githubusercontent.com/129328983/232885220-5fefe522-1223-42ad-b4d1-b1241a06c696.png)

#### Part IV

![image](https://user-images.githubusercontent.com/129328983/232882288-9a4b4d9e-d237-48a6-b29d-7e304910212b.png)

#### Part V

![image](https://user-images.githubusercontent.com/129328983/232882370-882e695c-9bd6-480b-aff6-24fda0613375.png)

#### Part VI

![image](https://user-images.githubusercontent.com/129328983/232882520-6801ea66-49eb-425f-a1bc-d456ebef54b5.png)

#### Part VII

![image](https://user-images.githubusercontent.com/129328983/232882593-d7883f20-e246-4495-a2b8-d39b7bf487b7.png)

![image](https://user-images.githubusercontent.com/129328983/232882662-cd6a8519-e9ef-425c-bb70-ecb5415aa5e7.png)


## Sec. V. Summary and Conclusions

#### Part I
From this correlational matrix I can see that the highest correlation of faces is in the high 80's and the lowest is much more spaced out. Most importantly, I notice a diagonal line going from zero onwards which shows a disproportionate correlation between faces of their own number with each other.

#### Part II
In these four pictures I can clearly see that the faces look very similar in the correlated category, and look almost opposite in the uncorrelated category. It was a necessity to remove the diagonal values or else it never would have worked properly. But, as seen by the correlation numbers, the highest is very high and the lowest is very low.

#### Part III
These results provide a wide array of data. It can be seen that some data ranges contain hotspots of high correlation, while others contain spots of low correlation. It is also important to note that even though the graph says from 1-11, that represents the first image not the second as Python likes to count from zero but it has been adjusted.

#### Part IV
These eigenvectors range from interesting to creepy. They are a generalization of faces, and act almost like a base template for the rest of the faces to have variance from. The largest vector (top left) most clearly resembles a real face which makes sense given it is the strongest value.

#### Part V
There is not a lot of data that can be drawn from these Principle components, I tried to visualize the data in a grayscale but there is no visible pattern. 

#### Part VI
It is very interesting to see that the SVD's first principle component and the strongest eigenvector share a very similar value. With an incredibly small difference (1*10^-15) it shows both these methods are strong ways to manipulate the data.

#### Part VII
Finally, I can see from the first six principle components that the first has the highest variance. Yet, as they progress the SVD modes become way less in variance. Also, from the plotting I can notice that the principle components look very similar to the eigenvectors except they are the inverted in color.
