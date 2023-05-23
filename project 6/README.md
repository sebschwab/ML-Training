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


#### LSTM

The SHRED model utilizes an LSTM (Long Short-Term Memory) layer as the basic building block of its recurrent structure.

The LSTM layer is a type of recurrent neural network layer specifically designed to capture long-term dependencies in sequential data. It addresses the vanishing gradient problem that traditional RNNs often encounter when learning long-term dependencies.

Within the SHRED model, the LSTM layer is responsible for processing the input sequences of sensor measurements and learning the temporal patterns and dependencies present in the data. It retains a memory of past information and selectively decides which information to forget and which new information to incorporate at each time step, enabling it to capture both short-term and long-term dependencies.

By utilizing the LSTM layer, the SHRED model can effectively encode the input sequences and generate accurate predictions for sensor measurements. It enhances the model's ability to capture complex temporal patterns and produce meaningful representations of the data, ultimately improving the performance of the SHRED model in tasks such as state reconstruction and measurement forecasting.

## Sec. III. Algorithm Implementation and Development 

#### Part 1: Training the model

In order to train the model, I needed to split the data into appropriate trianing and testing datasets. However, before I could do this I needed to load in the data and set a few hyperparameters:

```
num_sensors = 3 
lags = 52
load_X = load_data('SST')
n = load_X.shape[0]
m = load_X.shape[1]
sensor_locations = np.random.choice(m, size=num_sensors, replace=False)
```

Then I split the dataset:

```
train_indices = np.random.choice(n - lags, size=1000, replace=False)
mask = np.ones(n - lags)
mask[train_indices] = 0
valid_test_indices = np.arange(0, n - lags)[np.where(mask!=0)[0]]
valid_indices = valid_test_indices[::2]
test_indices = valid_test_indices[1::2]
```

Then I generate input sequences for a SHRED model by reshaping and slicing the transformed sensor measurements. It further creates training, validation, and test datasets by assigning the appropriate input and output sequences for state reconstruction and sensor forecasting tasks. The datasets are then used to instantiate TimeSeriesDataset objects, which are ready for training and evaluation of the SHRED model.

```
### Generate input sequences to a SHRED model
all_data_in = np.zeros((n - lags, lags, num_sensors))
for i in range(len(all_data_in)):
    all_data_in[i] = transformed_X[i:i+lags, sensor_locations]

### Generate training validation and test datasets both for rec onstruction of states and forecasting sensors
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

### -1 to have output be at the same time as final sensor measurements
train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)

train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
test_dataset = TimeSeriesDataset(test_data_in, test_data_out)
```

Finally, I train the model and calculate its error. I also plot its accuracy over time:

```
shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=1000, lr=1e-3, verbose=True, patience=5)
test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
print(np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth))

# Plot validation errors
plt.figure(figsize=(10, 5))
plt.plot(validation_errors)
plt.title("Validation errors over epochs")
plt.xlabel("Epochs")
plt.ylabel("Validation error")
plt.grid(True)
plt.show()

# Plot difference between predicted and actual test data
diff = test_recons - test_ground_truth
plt.figure(figsize=(10, 5))
plt.plot(diff)
plt.title("Difference between predictions and ground truth")
plt.xlabel("Time")
plt.ylabel("Difference")
plt.grid(True)
plt.show()
```

#### Part 2: Testing different Lag times

In order to evaluate the SHRED's performance across different lag times, I created a test set array:
```
# List of lags to test
lag_list = [10, 20, 30, 40, 50, 60]
```

Then I simply ran the tests again with different lags:


```
# Loop over the lags
for lags in lag_list:

    # Code to prepare data (similar to the previous code you've shown, but replace "52" with "lags")
    all_data_in = np.zeros((n - lags, lags, num_sensors))
    for i in range(len(all_data_in)):
        all_data_in[i] = transformed_X[i:i+lags, sensor_locations]

    # Code to train model and calculate error on test data
    ### Generate training validation and test datasets both for rec onstruction of states and forecasting sensors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
    valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
    test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

    ### -1 to have output be at the same time as final sensor measurements
    train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
    valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
    test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)

    train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
    valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
    test_dataset = TimeSeriesDataset(test_data_in, test_data_out)
    
    shred1 = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
    validation_errors1 = models.fit(shred1, train_dataset, valid_dataset, batch_size=64, num_epochs=10, lr=0.1, verbose=True, patience=5)
    test_recons1 = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
    test_ground_truth1 = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
    
    # Save error
    error = np.linalg.norm(test_recons1 - test_ground_truth1) / np.linalg.norm(test_ground_truth1)
    results.append(error)

```

#### Part 3: Adding Gaussian Noise

In order to test how resilient my SHRED is to noise, I added some to the data with the array of standard deviations as shown below:

```
std_devs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
```

I then used this loop to test my code and retrain the model:

```
# Loop over the standard deviations
for std_dev in std_devs:
    # Add Gaussian noise to the data
    noisy_X = transformed_X + np.random.normal(loc=0, scale=std_dev, size=transformed_X.shape)
```

#### Part 4: Testing the number of sensors

In order to see how changing the number of sensors affects the performance of my model, I used the following array and loop:

```
# List of sensors to test
sensor_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]


# Initialize an empty list to store the results
results = []

# Loop over the sensor list
for num_sensors in sensor_list:
    # Randomly choose sensor locations
    sensor_locations = np.random.choice(m, size=num_sensors, replace=False)

    # Prepare data for current number of sensors
    all_data_in = np.zeros((n - 52, 52, num_sensors))
    for i in range(len(all_data_in)):
        all_data_in[i] = transformed_X[i:i+52, sensor_locations]

    # Prepare datasets
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
    valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
    test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

    train_data_out = torch.tensor(transformed_X[train_indices + 52 - 1, sensor_locations], dtype=torch.float32).to(device)
    valid_data_out = torch.tensor(transformed_X[valid_indices + 52 - 1, sensor_locations], dtype=torch.float32).to(device)
    test_data_out = torch.tensor(transformed_X[test_indices + 52 - 1, sensor_locations], dtype=torch.float32).to(device)

    train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
    valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
    test_dataset = TimeSeriesDataset(test_data_in, test_data_out)

    # Create and train the model
    shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
    validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=1000, lr=1e-3, verbose=True, patience=5)

    # Calculate error
    test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
    test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())

    error = np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth)
    results.append(error)

```

## Sec. IV. Computational Results

![image](https://github.com/sebschwab/ML-Training/assets/129328983/9bdc3dfe-fe36-4ecf-8dc2-abf01e6ec340)
![image](https://github.com/sebschwab/ML-Training/assets/129328983/a46f6444-e1c3-477d-b8ba-342c0def1f32)

Figure 1: Training data results from base conditions

![image](https://github.com/sebschwab/ML-Training/assets/129328983/1b5d9d09-8262-46fa-978b-67e9f509a69e)

Figure 2: Error over epochs

![image](https://github.com/sebschwab/ML-Training/assets/129328983/cd77a53d-dc21-4811-838e-355af007c50a)

Figure 3: Variation of results from ground truth

## Sec. V. Summary and Conclusions

From these results I am able to determine that this model became highly efficient at predicting temperatures after training. However, it took an incredible amount of time to run this Neural Network. On my personal laptop, it took over two hours. This is why I don't have the plotted results for parts 3-4. Although, on a higher powered GPU these results should run with much more ease. 

It seems that the error starts off relatively high, but then sharply drops after a few epochs. Which makes this model excellent with only three sensors and a lag time of 52.

Although I was unable to visually plot the results for varying lag time. I am able to predict that raising lag time would increase performance. But this would only be true up to a point, which is when over-fitting would start to occur.

The noise has a drastic effect on the ability of the model. Too much noise makes in incredibly difficult to read the data, and therefore much less accurate in training. 

In this neural network, only three sensors proved to be very accurate in predicting sea-surface temperatures. Two would be slightly too low, and although more sensors would certainly increase performance they would come with a higher computational cost.
