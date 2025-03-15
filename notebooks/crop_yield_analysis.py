#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crop Yield Analysis Script

This script analyzes agricultural data and predicts crop yield using Machine Learning models.
It performs exploratory data analysis on the crop_yield.csv dataset.

Author: FarmTech Solutions Team
Date: March 15, 2025
"""

# Import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Set plot style and display settings
def setup_environment():
    """Set up the environment for data analysis."""
    plt.style.use('seaborn-v0_8-whitegrid')
    # If the above style is not available, try a default style
    # plt.style.use('default')
    sns.set_palette('viridis')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 20)
    np.random.seed(42)  # Set random seed for reproducibility
    
    # Determine the correct images directory path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    global IMAGES_DIR
    IMAGES_DIR = os.path.join(project_dir, 'images')
    
    # Create images directory if it doesn't exist
    os.makedirs(IMAGES_DIR, exist_ok=True)
    print(f"Images will be saved to: {IMAGES_DIR}")

def load_data(file_path='../data/crop_yield.csv'):
    """
    Load the dataset from the specified file path.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    print(f"Loading data from {file_path}...")
    
    # Try different path options if the file is not found
    if not os.path.exists(file_path):
        # Try without the '../' prefix
        alt_path = file_path.replace('../', '')
        if os.path.exists(alt_path):
            file_path = alt_path
            print(f"Using alternative path: {file_path}")
        # Try with absolute path
        elif os.path.exists('/Users/gabriel/www/fiap/year-01/fase-05/farm-tech-solutions-v5/data/crop_yield.csv'):
            file_path = '/Users/gabriel/www/fiap/year-01/fase-05/farm-tech-solutions-v5/data/crop_yield.csv'
            print(f"Using absolute path: {file_path}")
        # Try with current directory
        elif os.path.exists('crop_yield.csv'):
            file_path = 'crop_yield.csv'
            print(f"Using file in current directory: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def display_dataset_info(df):
    """
    Display general information about the dataset.
    
    Args:
        df (pandas.DataFrame): Dataset to analyze
    """
    print("\n" + "="*50)
    print("DATASET INFORMATION")
    print("="*50)
    
    # Display dataset shape
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Display column names
    print("\nColumn names:")
    for col in df.columns:
        print(f"- {col}")
    
    # Display data types
    print("\nData types:")
    print(df.dtypes)
    
    # Display first few rows
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    # Display basic statistical summary
    print("\nStatistical summary of numerical columns:")
    print(df.describe())
    
    # Display unique values in categorical columns
    print(f"\nUnique crop types: {df['Crop'].nunique()}")
    print("\nList of unique crops:")
    print(df['Crop'].unique())

def analyze_missing_values(df):
    """
    Analyze missing values in the dataset.
    
    Args:
        df (pandas.DataFrame): Dataset to analyze
        
    Returns:
        pandas.DataFrame: DataFrame with missing values information
    """
    print("\n" + "="*50)
    print("MISSING VALUES ANALYSIS")
    print("="*50)
    
    # Check for missing values
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    # Create a DataFrame to display missing values information
    missing_info = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage (%)': missing_percentage
    })
    
    print("Missing values per column:")
    print(missing_info)
    
    # Visualize missing values if any
    if missing_values.sum() > 0:
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cmap='viridis', cbar=False, yticklabels=False)
        plt.title('Missing Values Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, 'missing_values_heatmap.png'))
        plt.close()
        print(f"Missing values heatmap saved to '{os.path.join(IMAGES_DIR, 'missing_values_heatmap.png')}'")
    else:
        print("No missing values found in the dataset.")
    
    return missing_info

def analyze_data_distribution(df):
    """
    Analyze the distribution of variables in the dataset.
    
    Args:
        df (pandas.DataFrame): Dataset to analyze
    """
    print("\n" + "="*50)
    print("DATA DISTRIBUTION ANALYSIS")
    print("="*50)
    
    # Distribution of the target variable (Yield)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Yield'], kde=True)
    plt.title('Distribution of Crop Yield')
    plt.xlabel('Yield')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(IMAGES_DIR, 'yield_distribution.png'))
    plt.close()
    print(f"Yield distribution plot saved to '{os.path.join(IMAGES_DIR, 'yield_distribution.png')}'")
    
    # Distribution of numerical features
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numerical_cols.remove('Yield')  # Remove target variable
    
    # Create histograms for numerical features
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(2, 2, i)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'numerical_features_distribution.png'))
    plt.close()
    print(f"Numerical features distribution plots saved to '{os.path.join(IMAGES_DIR, 'numerical_features_distribution.png')}'")
    
    # Create box plots for numerical features
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(2, 2, i)
        sns.boxplot(y=df[col])
        plt.title(f'Box Plot of {col}')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'numerical_features_boxplots.png'))
    plt.close()
    print(f"Numerical features box plots saved to '{os.path.join(IMAGES_DIR, 'numerical_features_boxplots.png')}'")
    
    # Yield distribution by crop type
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Crop', y='Yield', data=df)
    plt.title('Yield Distribution by Crop Type')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'yield_by_crop_type.png'))
    plt.close()
    print(f"Yield distribution by crop type plot saved to '{os.path.join(IMAGES_DIR, 'yield_by_crop_type.png')}'")
    
    return numerical_cols

def analyze_correlations(df, numerical_cols):
    """
    Analyze correlations between variables in the dataset.
    
    Args:
        df (pandas.DataFrame): Dataset to analyze
        numerical_cols (list): List of numerical column names
    """
    print("\n" + "="*50)
    print("CORRELATION ANALYSIS")
    print("="*50)
    
    # Calculate correlation matrix for numerical features
    numerical_df = df.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = numerical_df.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'correlation_matrix.png'))
    plt.close()
    print(f"Correlation matrix heatmap saved to '{os.path.join(IMAGES_DIR, 'correlation_matrix.png')}'")
    
    # Correlation with target variable (Yield)
    target_correlation = correlation_matrix['Yield'].sort_values(ascending=False)
    print("Correlation with Yield (target variable):")
    print(target_correlation)
    
    # Scatter plots of features vs. target
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(2, 2, i)
        sns.scatterplot(x=df[col], y=df['Yield'], alpha=0.6)
        plt.title(f'{col} vs. Yield')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'features_vs_yield.png'))
    plt.close()
    print(f"Features vs. yield scatter plots saved to '{os.path.join(IMAGES_DIR, 'features_vs_yield.png')}'")

def summarize_findings(df, missing_info):
    """
    Summarize key findings from the exploratory data analysis.
    
    Args:
        df (pandas.DataFrame): Dataset analyzed
        missing_info (pandas.DataFrame): Missing values information
    """
    print("\n" + "="*50)
    print("SUMMARY OF FINDINGS")
    print("="*50)
    
    print("Based on the exploratory data analysis, we can summarize the following key findings:")
    
    # Dataset Overview
    print("\n1. Dataset Overview:")
    print(f"   - The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
    print(f"   - It includes information about {df['Crop'].nunique()} different crop types.")
    print("   - The dataset contains environmental factors like precipitation, humidity, and temperature.")
    
    # Missing Values
    print("\n2. Missing Values:")
    if missing_info['Missing Values'].sum() > 0:
        print(f"   - The dataset contains {missing_info['Missing Values'].sum()} missing values.")
        cols_with_missing = missing_info[missing_info['Missing Values'] > 0].index.tolist()
        print(f"   - Columns with missing values: {', '.join(cols_with_missing)}")
    else:
        print("   - The dataset does not contain any missing values.")
    
    # Data Distribution
    print("\n3. Data Distribution:")
    print(f"   - The target variable 'Yield' ranges from {df['Yield'].min():.2f} to {df['Yield'].max():.2f}.")
    print(f"   - Mean yield across all crops: {df['Yield'].mean():.2f}")
    print(f"   - Crop with highest average yield: {df.groupby('Crop')['Yield'].mean().idxmax()}")
    print(f"   - Crop with lowest average yield: {df.groupby('Crop')['Yield'].mean().idxmin()}")
    
    # Correlation Analysis
    print("\n4. Correlation Analysis:")
    numerical_df = df.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = numerical_df.corr()
    target_correlation = correlation_matrix['Yield'].sort_values(ascending=False)
    
    # Get top correlations (excluding self-correlation)
    top_positive = target_correlation[1:4]  # Skip the first one (self-correlation)
    top_negative = target_correlation.tail(3)
    
    print("   - Top positive correlations with Yield:")
    for col, corr in top_positive.items():
        print(f"     * {col}: {corr:.3f}")
    
    print("   - Top negative correlations with Yield:")
    for col, corr in top_negative.items():
        print(f"     * {col}: {corr:.3f}")
    
    # Next Steps
    print("\n5. Next Steps:")
    print("   - Perform clustering analysis to identify patterns in crop productivity.")
    print("   - Develop predictive models to forecast crop yields based on environmental factors.")
    print("   - Evaluate different machine learning algorithms for yield prediction.")

def main():
    """Main function to run the entire analysis."""
    print("="*50)
    print("CROP YIELD ANALYSIS")
    print("="*50)
    
    # Set up the environment
    setup_environment()
    
    # Load the dataset
    df = load_data()
    if df is None:
        return
    
    # Display dataset information
    display_dataset_info(df)
    
    # Analyze missing values
    missing_info = analyze_missing_values(df)
    
    # Analyze data distribution
    numerical_cols = analyze_data_distribution(df)
    
    # Analyze correlations
    analyze_correlations(df, numerical_cols)
    
    # Summarize findings
    summarize_findings(df, missing_info)
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    print(f"All visualizations have been saved to the '{IMAGES_DIR}' directory.")

if __name__ == "__main__":
    main()
