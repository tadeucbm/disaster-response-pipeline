# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Function to load the data and return a dataframe merged
    Input: 
        messages_filepath - csv file from messages
        categories_filepath - csv file from categories
    Output:
        dataframe merged (messages + categories)
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id', how='left')
    
    return df


def clean_data(df):
    """
    Function to make all data cleaning
    Input:
        df - dataframe merged
    Output:
        A dataframe that pass from theses steps:
            - clean of categories - transform in 0 and 1
            - drop categorical data
            - drop duplicates
    """
    
    categories = df['categories'].str.split(';', expand=True) # df with categories
    categories.columns = [name[0] for name in categories.loc[0, :].str.split('-')] # set column names
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
        # convert column from string to numeric
        categories[column] = categories[column].astype('int64')
    # drop the original categories column from `df`
    df = pd.merge(df, categories, left_index=True, right_index=True).drop('categories', axis=1)
    # drop duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """
    Function the save the final df in a database
    Input:
        df - the dataframe cleaned
        database_filename - database destination
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, index=False, if_exists='replace')  

def main():
    """
    Function main to make all data processing
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()