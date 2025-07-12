import pandas as pd
import numpy as np

# Coefficients obtained from the LINEST output
coefficients = [-0.104674002,0.17494234,0.22843003,0.26175687,-0.0478894,0.1239252,-0.0535111,0.58261092]

# Load validation dataset
df_validation = pd.read_excel("Validation_Set_MinMax.xlsx")
columns_of_interest_validation = df_validation.iloc[:, 9:17]
index_flood_column_validation = df_validation.iloc[:, 17]

# Multiply each predictor variable by its corresponding coefficient and sum up the products
y_pred_validation = np.dot(columns_of_interest_validation, coefficients)

# Load actual target values for validation dataset
y_val_actual = index_flood_column_validation.to_numpy()

# Calculate squared differences
squared_diff = (y_val_actual - y_pred_validation) ** 2

# Calculate mean squared difference
mean_squared_diff = np.mean(squared_diff)

# Calculate RMSE for validation dataset
rmse_validation = np.sqrt(mean_squared_diff)
print("RMSE for validation dataset:", rmse_validation)

# Load test dataset
df_test = pd.read_excel("Test_Set_MinMax.xlsx")
columns_of_interest_test = df_test.iloc[:, 9:17]
index_flood_column_test = df_test.iloc[:, 17]

# Apply the model (coefficients) to the test dataset to obtain predicted values
y_pred_test = np.dot(columns_of_interest_test, coefficients)

# Load actual target values for test dataset
y_test_actual = index_flood_column_test.to_numpy()

# Calculate squared differences
squared_diff_test = (y_test_actual - y_pred_test) ** 2

# Calculate mean squared difference
mean_squared_diff_test = np.mean(squared_diff_test)

# Calculate RMSE for test dataset
rmse_test = np.sqrt(mean_squared_diff_test)
print("RMSE for test dataset:", rmse_test)