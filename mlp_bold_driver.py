import numpy as np
import pandas as pd
from numpy.random import default_rng
import matplotlib.pyplot as plt

rng = np.random.default_rng()
#generate biases for each node in hidden layer
def get_bias_or_get_weight(input_size):
    return rng.uniform((-2/input_size), (2/input_size))

# Define the neural network architecture parameters
input_size = int(input("how many predictors?"))  # Number of input features
# min_hidden_nodes = input_size // 2
# max_hidden_nodes = 2 * input_size
# Generate a random number between min_hidden_nodes and max_hidden_nodes
hidden_nodes = int(input("how many hidden nodes?"))

# Round the value to an integer (if needed)
hidden_nodes = round(hidden_nodes)
output_nodes = 1  # Number of outputs

# Learning rate
p = 0.1 

#generate bias for output layer 
outputBias = get_bias_or_get_weight(input_size)   

'''
creates the input_ matrix where it gets all the input catchments and generates in the right dimensions for dot product so places a 1 in the first column to get a 1*9 input matrix
'''
def generate_input_matrix(selected_data_values):
    # Reshape the 1D array to a 2D array (assuming it's a row vector)
    selected_data_values_2d = selected_data_values.reshape(1, -1)
    # New column to add
    new_column = np.ones((selected_data_values_2d.shape[0], 1))
    # Concatenate the new column to the existing matrix
    input_matrix=np.concatenate((new_column,selected_data_values_2d),axis=1)
    return input_matrix

'''
creates the hidden layer matrix by making the first row the bias for each node with each column representing
a node so a column's first row is the node's bias and the subsequent rows in the column is the weight assigned from input to the hidden node in question
'''
def generate_hidden_matrix(input_size):
    # Use NumPy to generate the entire matrix in one go
    return np.random.uniform((-2/input_size), (2/input_size), (input_size+1, hidden_nodes))

# Initialize weights and biases
hidden_matrix = generate_hidden_matrix(input_size)

#use this function to work out weighted sum using dot product of two matrices
def get_weighted_sum_matrix(matrix_1,matrix_2):
   result = np.dot(matrix_1,matrix_2)
   return result

#function gets weighted sum of each hidden node and then works out sigmoid function on each to get 1*(hidden_nodes+1) matrix 
def get_sigmoid_matrix(weighted_sum_matrix):
    sigmoid_matrix = np.zeros(hidden_nodes+1)
    # New column to add
    sigmoid_matrix[0] = 1
    for i in range(1,hidden_nodes+1):
        #sigmoid matrix is updated with values from the 2nd column onwards 
        #weighted sum matrix uses zero indexing therefore need to use i-1 to make sure i matches to right index
        sigmoid_matrix[i] = 1 / (1 + np.exp(-weighted_sum_matrix[0][i-1]))
    return sigmoid_matrix

'''
creates the output matrix which is the bias on output node (row 1) and weights from each hidden node to output
so you get a (hidden_nodes+1 * 1) matrix
'''
def generate_output_matrix():
    output_matrix = np.zeros((hidden_nodes+1, 1))
    for j in range(hidden_nodes+1):
        output_matrix[j][0] = get_bias_or_get_weight(input_size)
    return output_matrix

output_matrix = generate_output_matrix()

#apply sigmoid function to final weighted sum to get predicted index flood
def get_index_flood_predicted(x):
    index_flood_predicted = 1 / (1 + np.exp(-x))
    return index_flood_predicted

#gets the output delta needed to update the weight values 
def get_output_delta(index_flood_actual,x):
    #x is the predicted index flood
    output_delta = (index_flood_actual - x)*(x*(1-x))
    return output_delta[0]

#work out the delta values for each hidden node and use to update weights later on
def get_hidden_delta(output_delta, hidden_sigmoid_matrix, output_matrix):
    # Extract weights from hidden layer to output node
    hidden_weights = output_matrix[1:, 0]

    # Calculate hidden delta in a vectorized way
    #∂j=Wj,o*∂o*Ui(1-Ui)
    hidden_delta = hidden_weights * output_delta * hidden_sigmoid_matrix[1:] * (1 - hidden_sigmoid_matrix[1:])
    return hidden_delta

#initialise momentum matrices to store the change in weights and biases for momentum
output_momentum = np.zeros_like(output_matrix)
hidden_momentum = np.zeros_like(hidden_matrix)
# Momentum coefficient
alpha = 0.9 

#updates all the weights and biases using the delta values calculated
def update_weights_and_biases(output_matrix,hidden_matrix,output_delta,hidden_delta,output_momentum,hidden_momentum):

    # Update bias for output layer with momentum
    output_bias_update = p * output_delta + alpha * output_momentum[0,0]
    output_matrix[0,0] += output_bias_update
    output_momentum[0,0] = output_bias_update
    
    #update the weights from hidden nodes to output
    for i in range(1,hidden_nodes+1):
        weight_update_output = p * output_delta * hidden_sigmoid_matrix[i] + alpha * output_momentum[i,0]
        output_matrix[i,0] += weight_update_output
        output_momentum[i,0] = weight_update_output

    for i in range(hidden_nodes):
        bias_update_hidden = p * hidden_delta[i] + alpha * hidden_momentum[0,i]
        hidden_matrix[0,i] += bias_update_hidden
        hidden_momentum[0,i] = bias_update_hidden
        for j in range(1,input_size+1):
            weight_update_hidden = p * hidden_delta[i] * input_matrix[0,j] + alpha * hidden_momentum[j,i]
            hidden_matrix[j,i] += weight_update_hidden
            hidden_momentum[j,i] = weight_update_hidden 
    return hidden_matrix, output_matrix

# Mean squared error function
def mean_squared_error(predicted, actual):
    return np.mean((actual - predicted) ** 2)

# MSE history
mse_history_training = []
mse_history_validation = []

# Number of epochs
epochs = int(input("Enter the number of epochs: "))
epochs_validation = []

# Load training dataset
df_training = pd.read_excel("Training_Set_MinMax.xlsx")
columns_of_interest_training = df_training.columns[9:17]
index_flood_column_training = df_training.columns[17]

# Load validation dataset
df_validation = pd.read_excel("Validation_Set_MinMax.xlsx")
columns_of_interest_validation = df_validation.columns[9:17]
index_flood_column_validation = df_validation.columns[17]


for epoch in range(epochs):
    total_error_training = 0
    total_error_validation = 0

    # Ensure there are at least two previous epochs to compare MSEs
    if epoch >= 2000 and epoch % 2000 == 0:
        # Calculate the difference in MSE between the last two recorded epochs
        mse_diff = mse_history_training[-2] - mse_history_training[-1]  # Use -2 and -1 for the last two items in list
        
        # If the MSE decreased (improvement), consider increasing the learning rate
        if mse_diff > 0:
            p *= 1.05
            print(f"Increasing learning rate to {p} at epoch {epoch}")
        # If the MSE increased (worsening), decrease the learning rate
        elif mse_diff < 0:
            p *= 0.7
            print(f"Decreasing learning rate to {p} at epoch {epoch}")
        else:
            p = p

    if epoch >= 100 and epoch % 100 == 0:
        df = df_validation
        columns_of_interest = columns_of_interest_validation
        index_flood_column = index_flood_column_validation
        epochs_validation.append(epoch)
    else:
        df = df_training
        columns_of_interest = columns_of_interest_training
        index_flood_column = index_flood_column_training

    for i in range(len(df)):
        # Extracting row-wise data
        selected_data_values = np.array(df.loc[i, columns_of_interest])
        index_flood_actual = df.loc[i, index_flood_column]
        # Generate matrices for input and perform forward propagation
        input_matrix = generate_input_matrix(selected_data_values)
        hidden_sigmoid_matrix = get_sigmoid_matrix(get_weighted_sum_matrix(input_matrix, hidden_matrix))
        output_weighted_sum = get_weighted_sum_matrix(hidden_sigmoid_matrix, output_matrix)
        index_flood_predicted = get_index_flood_predicted(output_weighted_sum)

        # Calculate error
        error_training = mean_squared_error(index_flood_predicted, index_flood_actual)
        total_error_training += error_training
        # Update weights and biases based on backpropagation
        output_delta = get_output_delta(index_flood_actual,index_flood_predicted)
        hidden_delta = get_hidden_delta(output_delta,hidden_sigmoid_matrix,output_matrix)
        update_weights_and_biases(output_matrix,hidden_matrix,output_delta,hidden_delta,output_momentum,hidden_momentum)

        if df.equals(df_training):
                avg_epoch_error = total_error_training / len(df)
                mse_history_training.append(avg_epoch_error)
        # Average error for the epoch
        if df.equals(df_validation):
            # Calculate error
            error_training = mean_squared_error(index_flood_predicted, index_flood_actual)
            total_error_validation += error_training

    if df.equals(df_validation):   
        avg_epoch_error = total_error_validation/ len(df)
        mse_history_validation.append(avg_epoch_error)
        print(f"Epoch {epoch+1}: Validation MSE = {avg_epoch_error}, {hidden_nodes} hidden nodes")

# Plot MSE history for validation data
# Assuming validation MSE is recorded every 100 epochs
plt.figure(figsize=(10, 6))
plt.plot(epochs_validation, mse_history_validation, label='MSE Validation', color='red')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('MSE over Epochs - Validation Data Bold Driver')
plt.legend()
plt.show()

# Convert MSE values to RMSE values
rmse_array = np.sqrt(mse_history_validation)

# Print or use the RMSE values as needed
min_rmse = np.min(rmse_array)
index = np.where(rmse_array == min_rmse)[0]
optimal_epoch = epochs_validation[index[0]]
print("RMSE:",min_rmse, optimal_epoch)

# Load test dataset
df_test = pd.read_excel("Test_Set_MinMax.xlsx")
columns_of_interest_test = df_test.columns[9:17]
index_flood_column_test = df_test.columns[17]

# Lists to store predicted and actual values
modelled_test = []
observed_test = []

# Iterate over the test data
for i in range(len(df_test)):
    # Extracting row-wise data
    selected_data_values = np.array(df_test.loc[i, columns_of_interest_test])
    index_flood_actual = df_test.loc[i, index_flood_column_test]

    # Generate matrices for input and perform forward propagation
    input_matrix = generate_input_matrix(selected_data_values)
    hidden_sigmoid_matrix = get_sigmoid_matrix(get_weighted_sum_matrix(input_matrix, hidden_matrix))
    output_weighted_sum = get_weighted_sum_matrix(hidden_sigmoid_matrix, output_matrix)
    index_flood_predicted = get_index_flood_predicted(output_weighted_sum)

    # Store predicted and actual values
    modelled_test.append(index_flood_predicted)
    observed_test.append(index_flood_actual)

# Plot predicted and actual values
plt.figure(figsize=(10, 6))
plt.plot(range(len(df_test)), modelled_test, label='Predicted', color='blue')
plt.plot(range(len(df_test)), observed_test, label='Actual', color='red')
plt.xlabel('Sample')
plt.ylabel('Index Flood Value')
plt.title('Predicted vs Actual - Test Data Bold Driver ')
plt.legend()
plt.show()

# Create scatter plot
plt.scatter(observed_test, modelled_test)
plt.title('MLP Scatter Plot')
plt.xlabel('Observed')
plt.ylabel('Modelled')
plt.grid(True)
plt.show()

def correlation_coefficient(x, y):
    """
    Calculate the correlation coefficient between two arrays of data.
    
    Parameters:
        x (array): First array of data.
        y (array): Second array of data.
        
    Returns:
        float: Correlation coefficient between the two arrays.
    """
    # Convert input arrays to numpy arrays if they are not already
    x = np.array(x)
    y = np.array(y)
    
    # Calculate means of x and y
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Calculate covariance and standard deviations
    covariance = np.mean((x - mean_x) * (y - mean_y))
    std_dev_x = np.std(x)
    std_dev_y = np.std(y)
    
    # Calculate correlation coefficient
    correlation = covariance / (std_dev_x * std_dev_y)
    
    return correlation

print("Correlation coefficient:", correlation_coefficient(observed_test, modelled_test))