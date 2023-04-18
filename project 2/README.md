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

## Sec. III. Algorithm Implementation and Development 

#### Part 1

To begin, I created a correlational matrix of the first 100 photos and plotted using the pcolor attribute in matplot.

'# Extract the first 100 images from the matrix X
X_100 = X[:, :100]

# Compute the correlation matrix
C = np.dot(X_100.T, X_100)

# Plot the correlation matrix
plt.pcolor(C)
plt.colorbar()
plt.title('Correlation Matrix of the First 100 Images')
plt.xlabel('Image Index')
plt.ylabel('Image Index')
plt.show()'

## Sec. IV. Computational Results
## Sec. V. Summary and Conclusions
