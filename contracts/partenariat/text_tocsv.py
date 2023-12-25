import os
import pandas as pd
import re
from glob import glob

# Specify the folder containing the .txt files
folder_path = "./"

# Get a list of all .txt files in the folder
txt_files = glob(os.path.join(folder_path, '*.txt'))

# Create an empty list to store DataFrames for all data and articles
all_data_dfs = []
article_dfs = []

# Iterate over each .txt file
for txt_file in txt_files:
    # Read the text file
    with open(txt_file, 'r', encoding='utf-8') as file:
        lines = file.read().split('\n')

    # Remove non-printable characters
    lines = [re.sub(r'[﻿]', '', line) for line in lines]

    # Create a DataFrame from the text file
    df = pd.DataFrame({'Content': lines})
    
    # Remove empty rows
    df = df[df['Content'].str.strip() != '']

    # Reindex the DataFrame
    df.index = range(1, len(df) + 1)

    # Append the DataFrame to the list of all data
    all_data_dfs.append(df)

    # Extract paragraphs starting with "ARTICLE" and ending before the next "ARTICLE"
    article_df = pd.DataFrame(columns=['Content'])  # Initialize an empty DataFrame for articles
    article_started = False
    current_article = ""

    for index, row in df.iterrows():
        line = row['Content']
        if line.upper().startswith('ARTICLE'):
            if article_started:
                # Save the current article and start a new one
                article_df = pd.concat([article_df, pd.DataFrame({'Content': [current_article]})], ignore_index=True)
                current_article = ""
            article_started = True
        if article_started:
            current_article += line + '\n'

    # Append the last article if any
    if current_article:
        article_df = pd.concat([article_df, pd.DataFrame({'Content': [current_article]})], ignore_index=True)

    # Append the DataFrame of articles to the list
    if not article_df.empty:
        article_dfs.append(article_df)

# Concatenate all DataFrames in the list into a single DataFrame for all data
all_data = pd.concat(all_data_dfs, ignore_index=True)

# Check if there are articles to concatenate
if article_dfs:
    # Concatenate all DataFrames in the list into a single DataFrame for articles
    all_articles = pd.concat(article_dfs, ignore_index=True)

    # Get the directory of one of the input files
    output_folder = os.path.dirname(txt_files[0])

    # Save the combined DataFrame for articles to a CSV file in the same folder
    article_output_path = os.path.join(output_folder, 'output_combined_articles.csv')
    all_articles.to_csv(article_output_path, index=True, encoding='utf-8')
    print(f"Articles saved to: {article_output_path}")
else:
    print("No articles found.")
