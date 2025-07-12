import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load your dataset
df = pd.read_excel('AI_Dataset.xlsx', engine='openpyxl')

# Define the list of descriptors to plot against the Index flood.
# Exclude 'Index flood' from the plotting list if it's included in the columns.
descriptors = df.columns[:-1]  # Assuming 'Index flood' is the last column

# Iterate through each descriptor and create a scatter plot against the Index flood
for descriptor in descriptors:
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    sns.scatterplot(data=df, x=descriptor, y='Index flood', color='blue', alpha=0.6)
    
    # Adding title and labels
    plt.title(f'Scatter Plot of {descriptor} vs. Index Flood')
    plt.xlabel(descriptor)
    plt.ylabel('Index Flood (m³/s)')
    
    # Show the plot
    plt.show()
