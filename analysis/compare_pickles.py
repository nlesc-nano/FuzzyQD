import pickle
import matplotlib.pyplot as plt
import numpy as np

def extract_and_plot(data1, data2):
    """
    Extract the relevant data and plot it.

    Parameters:
    data1: Content of the first pickle file.
    data2: Content of the second pickle file.
    """
    try:
        # Extract the second element (assumed to be the data to plot)
        data1_array = data1[1]
        data2_array = data2[1]

        # Ensure data is 2D for plotting
        if len(data1_array.shape) == 2 and data1_array.shape[0] == 1:
            data1_array = data1_array.squeeze(0)  # Remove singleton dimension
        if len(data2_array.shape) == 2 and data2_array.shape[0] == 1:
            data2_array = data2_array.squeeze(0)  # Remove singleton dimension

        # Generate x-axis
        x_axis = np.arange(data1_array.shape[0])

        # Plot data1 and data2
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.title("Pickle File 1")
        plt.plot(x_axis, data1_array, label="Data 1")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.title("Pickle File 2")
        plt.plot(x_axis, data2_array, label="Data 2")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Could not plot the data: {e}")

def load_and_compare(file1, file2):
    """
    Load two pickle files, extract relevant data, and plot them.

    Parameters:
    file1 (str): Path to the first pickle file.
    file2 (str): Path to the second pickle file.
    """
    try:
        with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
            data1 = pickle.load(f1)
            data2 = pickle.load(f2)

        print("File 1 Metadata (Element 0):", data1[0])
        print("File 2 Metadata (Element 0):", data2[0])
        print("File 1 Data (Element 1) Shape:", data1[1].shape)
        print("File 2 Data (Element 1) Shape:", data2[1].shape)

        extract_and_plot(data1, data2)

    except Exception as e:
        print(f"An error occurred: {e}")

# Replace these with your pickle file paths
file1 = 'file1.pkl'
file2 = 'file2.pkl'

load_and_compare(file1, file2)

