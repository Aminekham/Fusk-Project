import os
import pandas as pd

def clean_content_column(csv_path):
    df = pd.read_csv(csv_path)
    df['Content'] = df['Content'].replace(r'\{\d+\}', '...', regex=True)

    # Save the cleaned DataFrame back to the CSV file
    df.to_csv(csv_path, index=False)

def process_csv_files(root_directory='.'):
    # Iterate through all files in the directory and its subdirectories
    for foldername, subfolders, filenames in os.walk(root_directory):
        for filename in filenames:
            # Check if the file is a CSV file
            if filename.endswith('.csv'):
                csv_path = os.path.join(foldername, filename)
                clean_content_column(csv_path)

# Specify the root directory (change it to the directory containing your CSV files)
root_directory = './'

# Process all CSV files in the specified directory
process_csv_files(root_directory)
