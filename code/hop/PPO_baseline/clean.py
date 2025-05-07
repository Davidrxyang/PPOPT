import os
import pandas as pd

def modify_csv_files_in_directory(directory):
    """
    Modify all CSV files in the specified directory.
    For each file, set all negative reward values to 0.

    Args:
        directory (str): Path to the directory containing CSV files.
    """
    # Iterate through all files in the directory
    for file in os.listdir(directory):
        if file.endswith('.csv'):  # Process only CSV files
            file_path = os.path.join(directory, file)
            
            # Load the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            
            # Check if the 'reward' column exists
            if 'reward' in df.columns:
                # Set negative rewards to 0
                df['reward'] = df['reward'].apply(lambda x: -10 if x <= 0 else x)
                
                # Save the modified DataFrame back to the same file
                df.to_csv(file_path, index=False)
                print(f"Modified file: {file_path}")
            else:
                print(f"Skipping file (no 'reward' column): {file_path}")

# Example usage
current_directory = os.getcwd()  # Get the current working directory
modify_csv_files_in_directory(current_directory)