import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt

def plot_csv_files_from_directory(directory):
    """
    Plot rewards from all CSV files in a given directory.

    Args:
        directory (str): Path to the directory containing CSV files.
    """
    # Get all CSV files in the directory
    files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]
    
    if not files:
        raise ValueError("No CSV files found in the specified directory.")
    
    # Create labels based on file names (without extensions)
    labels = [os.path.splitext(os.path.basename(file))[0] for file in files]
    
    # Create a plot
    plt.figure(figsize=(10, 6))
    
    # Iterate through each file and label
    for file, label in zip(files, labels):
        # Read the CSV file
        data = pd.read_csv(file)
        
        # Calculate the rolling average
        rolling_avg = data['reward'].rolling(window=50).mean()
        
        # Plot the results
        plt.plot(data['episode'], rolling_avg, label=label)
    
    # Add labels and legend
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Rewards per Episode with Rolling Average = 50')
    plt.legend()
    
    # Show the plot
    plt.show()

# Example usage
if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Plot rewards from CSV files in a directory.")
    parser.add_argument("directory", type=str, help="Path to the directory containing CSV files.")

    # Parse arguments
    args = parser.parse_args()

    # Call the function with the provided directory
    plot_csv_files_from_directory(args.directory)