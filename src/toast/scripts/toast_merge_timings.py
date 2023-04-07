#!/usr/bin/env python3
"""
This script processes multiple timing.csv files located in the current directory and its subdirectories, performs various operations on their contents, and generates a merged CSV file.

The main steps performed by this script are:

1. Locate all timing.csv files in the current folder and its subfolders.
2. Load each CSV file into a DataFrame, keeping only the "Timer" and "Mean Time" columns.
3. Replace row names (index) ending with '_compiled', '_jax', or '_numpy' with '_kernel' in the individual DataFrames.
4. Merge all DataFrames using the intersection of all row names (Timer column) and keeping only the "Mean Time" columns, which are renamed to the name of each CSV file's containing folder.
5. Save the merged DataFrame to a new CSV file called merged_timings.csv in the current folder.

To use this script, simply run it in a Python environment with the required dependency (pandas) installed. The script will automatically locate timing.csv files, process them, generate the merged DataFrame and save the merged_timings.csv file.
"""
import os
import glob
import pandas as pd

def find_csv_files(folder, file_pattern):
    """
    Find all CSV files matching the specified pattern in the given folder and its subfolders.
    
    :param folder: The root folder to start searching for CSV files.
    :param file_pattern: The pattern of the CSV files to search for.
    :return: A list of file paths matching the specified pattern.
    """
    return glob.glob(os.path.join(folder, file_pattern), recursive=True)

def process_row_name(s):
    """
    Remove occurrences of '_compiled', '_numpy', '_jax', '(function) ', and '._exec'
    from the input string.If the resulting string contains '|dispatch|', add '_kernel'
    to the end of the string.

    :param s: Input string to process.
    :return: Processed string.
    """
    new_string = (s.replace('_compiled', '')
                   .replace('_numpy', '')
                   .replace('_jax', '')
                   .replace('(function) ', '')
                   .replace('._exec', ''))

    if '|dispatch|' in new_string:
        new_string += '_kernel'

    return new_string

def load_csv_files(file_paths):
    """
    Load CSV files from the given file paths, keeping only the specified columns
    and setting the index to the Timer column.

    :param file_paths: A list of file paths to load.
    :return: A list of DataFrames containing the loaded DataFrames.
    """
    dataframes = list()
    for file_path in file_paths:
        folder_name = os.path.basename(os.path.dirname(file_path))
        df = pd.read_csv(file_path, index_col="Timer", usecols=["Timer", "Mean Time"])

        # Cleanup/unifies the row names
        # Rename the 'Mean Time' column to the folder name
        df = df.rename(index=process_row_name, columns={"Mean Time": folder_name})

        dataframes.append(df)

    return dataframes

def merge_dataframes(dataframes):
    """
    Merge the given DataFrames using the union of all row names and keeping only
    the specified columns. The columns are intermeshed as best as possible based
    on their original order in the input CSV files.
    
    :param dataframes: A dictionary of DataFrames to merge.
    :return: A merged DataFrame with the union of all row names.
    """
    # Initialize the merged DataFrame with the first DataFrame
    merged_df = dataframes[0]

    # Merge the remaining DataFrames one-by-one
    for df in dataframes[1:]:
        merged_df = pd.merge(merged_df, df, left_index=True, right_index=True, how='outer')

    return merged_df

if __name__ == "__main__":
    folder = "."
    file_pattern = "**/timing.csv"
    csv_file_paths = find_csv_files(folder, file_pattern)
    dataframes = load_csv_files(csv_file_paths)
    merged_df = merge_dataframes(dataframes)

    print("Merged data:")
    print(merged_df)

    # Save the merged DataFrame to a CSV file
    output_file = "merged_timings.csv"
    merged_df.to_csv(output_file)
    print(f"\nMerged data saved to '{output_file}'")
