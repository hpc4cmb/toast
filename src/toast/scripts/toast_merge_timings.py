#!/usr/bin/env python3
"""
This script processes multiple timing.csv files located in the current directory and its subdirectories, performs various operations on their contents, and generates a merged CSV file.

The main steps performed by this script are:

1. Locate all timing.csv files in the current folder and its subfolders.
2. Load each CSV file into a DataFrame, keeping only the "Timer" and "Mean Time" columns.
3. Replace row names (index) ending with '_compiled', '_jax', or '_numpy' with '_kernel' in the individual DataFrames.
4. Merge all DataFrames using the intersection of all row names (Timer column) and keeping only the "Mean Time" columns, which are renamed to the name of each CSV file's containing folder.
5. Save the merged DataFrame to a new CSV file called merged_timings.csv in the current folder.
6. Similarly produce a merged_kernels_timing.csv file with the sum of the runtime for each kernel type, keeping only kernels that are GPU or for which more than one implementation was used

To use this script, simply run it in a Python environment with the required dependency (pandas) installed. The script will automatically locate timing.csv files, process them, generate the merged DataFrame and save the merged_timings.csv file.
"""
import glob
import os
import re

import numpy as np
import pandas as pd


def find_csv_files(folder, file_pattern):
    """
    Find all CSV files matching the specified pattern in the given folder and its subfolders.

    :param folder: The root folder to start searching for CSV files.
    :param file_pattern: The pattern of the CSV files to search for.
    :return: A list of file paths matching the specified pattern.
    """
    return glob.glob(os.path.join(folder, file_pattern), recursive=True)


def process_timer_path(s):
    """
    Process the input string and return three strings: simplified path, kernel type,
    and operation name.

    :param s: Input string to process.
    :return: A tuple containing the simplified path, kernel type, and operation name.
    """
    # Remove occurrences of '(function) ' and '._exec'
    simplified_path = s.replace("(function) ", "").replace("._exec", "")

    # Determine kernel type
    if s.endswith("_jax"):
        kernel_type = "JAX"
        simplified_path = simplified_path.replace("_jax", "")
    elif s.endswith("_compiled"):
        kernel_type = "COMPILED"
        simplified_path = simplified_path.replace("_compiled", "")
    elif s.endswith("_numpy"):
        kernel_type = "NUMPY"
        simplified_path = simplified_path.replace("_numpy", "")
    elif ("accel_data" in simplified_path) or ("INTERVALS_JAX" in simplified_path):
        kernel_type = "DATA_MOVEMENT"
    elif "|dispatch|" in simplified_path:
        kernel_type = "DEFAULT"
    else:
        kernel_type = None

    # Extract operation name
    operation_name = simplified_path.split("|")[-1]
    if kernel_type == "DATA_MOVEMENT":
        # Name clean-up specific to Jax data movement operations
        if operation_name == "INTERVALS_JAX.__init__":
            operation_name = "accel_data_update_device"
        elif operation_name == "INTERVALS_JAX.to_host":
            operation_name = "accel_data_update_host"

    # Add '_kernel' at the end of simplified_path if it contains '|dispatch|'
    if (kernel_type is not None) and (kernel_type != "DATA_MOVEMENT"):
        simplified_path += "_kernel"

    return simplified_path, kernel_type, operation_name


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

        # Process the index and extract simplified_path, kernel_type, and operation_name
        processed_indices = list(map(process_timer_path, df.index))
        simplified_paths, kernel_types, operation_names = zip(
            *processed_indices
        )  # unzip

        # Replace index with simplified_path
        df.index = simplified_paths

        # Add kernel_type and operation_name columns
        df["kernel_type"] = kernel_types
        df["operation_name"] = operation_names

        # Rename the 'Mean Time' column to the folder name
        df = df.rename(columns={"Mean Time": folder_name})

        dataframes.append(df)

    return dataframes


def combine_kernel_types(k1, k2):
    """
    Combine two kernel types into a single kernel type,
    returning 'MULTIPLE' if they differ.

    :param k1: The first kernel type (string or None).
    :param k2: The second kernel type (string or None).
    :return: The combined kernel type (string).
    """
    if (k1 is None) or (isinstance(k1, float) and np.isnan(k1)):
        return k2
    if (k2 is None) or (isinstance(k2, float) and np.isnan(k2)):
        return k1
    if k1 == k2:
        return k1
    return "MULTIPLE"


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
        merged_df = pd.merge(
            merged_df,
            df,
            left_index=True,
            right_index=True,
            how="outer",
            suffixes=("_1", "_2"),
        )

        # Combine kernel_type and operation_name columns
        merged_df["kernel_type"] = merged_df["kernel_type_1"].combine(
            merged_df["kernel_type_2"], combine_kernel_types
        )
        merged_df["operation_name"] = merged_df["operation_name_1"].combine_first(
            merged_df["operation_name_2"]
        )

        # Drop extra kernel_type and operation_name columns
        merged_df.drop(
            columns=[
                "kernel_type_1",
                "kernel_type_2",
                "operation_name_1",
                "operation_name_2",
            ],
            inplace=True,
        )

    return merged_df


def merge_kernel_rows(df):
    """
    Filters the input DataFrame to keep only rows where the `kernel_type` column is 'SEVERAL',
    drops the `kernel_type` column, and groups the DataFrame by the `operation_name` column,
    summing the values of rows that get collapsed together.

    :param df: Input DataFrame to filter and group.
    :return: A filtered and grouped DataFrame.
    """
    # Filter DataFrame to keep only rows where kernel_type has some GPU computation
    df_filtered = df[df["kernel_type"].isin(["MULTIPLE", "DATA_MOVEMENT", "JAX"])]
    # df_filtered = df[~df['kernel_type'].isin(['NUMPY', 'DEFAULT', None])] # TODO switch to this version once compiled becomes common

    # Drop the kernel_type column
    df_filtered = df_filtered.drop(columns=["kernel_type"])

    # Group by operation_name and sum the values of rows that get collapsed together
    # Some columns contains None/nan, hence the need for numeric_only=False
    df_grouped = df_filtered.groupby("operation_name").apply(
        lambda x: x.sum(numeric_only=False)
    )
    # Remove additional operation_name column added by the non-numeric-only summation
    df_grouped.drop(columns=["operation_name"], inplace=True)

    return df_grouped


if __name__ == "__main__":
    folder = "."
    file_pattern = "**/timing.csv"
    csv_file_paths = find_csv_files(folder, file_pattern)
    dataframes = load_csv_files(csv_file_paths)
    merged_df = merge_dataframes(dataframes)
    merged_kernel_df = merge_kernel_rows(merged_df)

    # Save the merged kernel data to a CSV file
    output_kernels_file = "merged_kernels_timings.csv"
    merged_kernel_df.to_csv(output_kernels_file)
    print(f"Merged kernel data saved to '{output_kernels_file}'")

    # Save the merged DataFrame to a CSV file
    merged_df = merged_df.drop(columns=["kernel_type", "operation_name"])
    output_file = "merged_timings.csv"
    merged_df.to_csv(output_file)
    print(f"Merged data saved to '{output_file}'")
