import pandas as pd
import re

def load_dataset(file_path):
    """
    Load the dataset from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file.
    
    Returns:
    pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)

def display_head(data, n=5):
    """
    Display the first few rows of the dataset.
    
    Parameters:
    data (pd.DataFrame): The dataset.
    n (int): Number of rows to display.
    """
    print(data.head(n))

def dataset_info(data):
    """
    Print the dataset summary.
    
    Parameters:
    data (pd.DataFrame): The dataset.
    """
    print(data.info())

def check_missing_values(data):
    """
    Check for missing values in the dataset.
    
    Parameters:
    data (pd.DataFrame): The dataset.
    
    Returns:
    pd.Series: Count of missing values per column.
    """
    return data.isnull().sum()

def fill_missing_values(data):
    """
    Fill missing values in the dataset.
    
    Parameters:
    data (pd.DataFrame): The dataset.
    
    Returns:
    pd.DataFrame: Dataset with missing values filled.
    """
    return data.fillna('', inplace=True)

def clean_text(text):
    """
    Clean the text data by removing punctuation and converting to lowercase.
    
    Parameters:
    text (str): The text to clean.
    
    Returns:
    str: Cleaned text.
    """
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

def preprocess_data(data):
    """
    Preprocess the dataset.
    
    Parameters:
    data (pd.DataFrame): The dataset.
    
    Returns:
    pd.DataFrame: Preprocessed dataset.
    """
    # Fill missing values
    fill_missing_values(data)
    
    # Clean text data
    data['cleaned_text'] = data['product'].apply(clean_text)
    
    # Convert categorical data to numerical
    categorical_columns = ['user id', 'product id']
    data = pd.get_dummies(data, columns=categorical_columns)
    
    # Convert 'rating' to binary target
    data['binary_rating'] = (data['rating'] >= 1000).astype(int)
    
    return data

def main():
    file_path = 'ecommerce_data.csv'
    data = load_dataset(file_path)
    
    display_head(data)
    dataset_info(data)
    
    missing_values = check_missing_values(data)
    print("Missing values:\n", missing_values)
    
    data = preprocess_data(data)
    display_head(data)

if __name__ == "__main__":
    main()
