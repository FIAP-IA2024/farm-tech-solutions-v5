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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Import libraries for supervised learning models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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

def preprocess_data_for_clustering(df):
    """
    Preprocess the data for clustering analysis.
    
    Args:
        df (pandas.DataFrame): Dataset to preprocess
        
    Returns:
        tuple: (preprocessed_df, scaled_features, feature_names, scaler)
    """
    print("\n" + "="*50)
    print("PREPROCESSING DATA FOR CLUSTERING")
    print("="*50)
    
    # Create a copy of the dataframe
    cluster_df = df.copy()
    
    # Remove non-numeric columns
    print("Removing non-numeric columns for clustering...")
    numeric_df = cluster_df.select_dtypes(include=['float64', 'int64'])
    feature_names = numeric_df.columns.tolist()
    print(f"Features used for clustering: {feature_names}")
    
    # Scale the features
    print("Scaling features...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(numeric_df)
    
    # Create a DataFrame with scaled features
    scaled_df = pd.DataFrame(scaled_features, columns=feature_names)
    
    print(f"Data preprocessed successfully. Shape: {scaled_df.shape}")
    return cluster_df, scaled_features, feature_names, scaler

def determine_optimal_clusters(scaled_features):
    """
    Determine the optimal number of clusters using Elbow Method and Silhouette Score.
    
    Args:
        scaled_features (numpy.ndarray): Scaled features for clustering
        
    Returns:
        int: Optimal number of clusters
    """
    print("\n" + "="*50)
    print("DETERMINING OPTIMAL NUMBER OF CLUSTERS")
    print("="*50)
    
    # Calculate inertia (within-cluster sum of squares) for different k values
    inertia = []
    silhouette_scores = []
    k_range = range(2, 11)  # Test from 2 to 10 clusters
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        inertia.append(kmeans.inertia_)
        
        # Calculate silhouette score
        labels = kmeans.labels_
        silhouette_avg = silhouette_score(scaled_features, labels)
        silhouette_scores.append(silhouette_avg)
        print(f"For n_clusters = {k}, the silhouette score is {silhouette_avg:.3f}")
    
    # Plot the Elbow Method
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Elbow Method
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertia, 'o-', markersize=8)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Silhouette Score
    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, 'o-', markersize=8)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'optimal_clusters.png'))
    plt.close()
    print(f"Optimal clusters plot saved to '{os.path.join(IMAGES_DIR, 'optimal_clusters.png')}'")
    
    # Find optimal k based on silhouette score (highest score)
    optimal_k = k_range[silhouette_scores.index(max(silhouette_scores))]
    print(f"Based on silhouette score, the optimal number of clusters is: {optimal_k}")
    
    # Check for elbow point (this is more subjective)
    print("Note: Visual inspection of the Elbow Method plot is also recommended.")
    
    return optimal_k

def perform_kmeans_clustering(cluster_df, scaled_features, optimal_k, feature_names):
    """
    Perform K-Means clustering with the optimal number of clusters.
    
    Args:
        cluster_df (pandas.DataFrame): Original dataframe
        scaled_features (numpy.ndarray): Scaled features for clustering
        optimal_k (int): Optimal number of clusters
        feature_names (list): Names of features used for clustering
        
    Returns:
        pandas.DataFrame: DataFrame with cluster assignments
    """
    print("\n" + "="*50)
    print(f"PERFORMING K-MEANS CLUSTERING WITH {optimal_k} CLUSTERS")
    print("="*50)
    
    # Apply K-Means with the optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    
    # Get cluster labels
    cluster_labels = kmeans.labels_
    
    # Add cluster labels to the original dataframe
    cluster_df['Cluster'] = cluster_labels
    
    # Display cluster information
    print("Cluster distribution:")
    cluster_counts = cluster_df['Cluster'].value_counts().sort_index()
    for cluster, count in cluster_counts.items():
        print(f"Cluster {cluster}: {count} samples ({count/len(cluster_df)*100:.1f}%)")
    
    # Calculate cluster centers in original feature space
    cluster_centers = kmeans.cluster_centers_
    
    # Create a DataFrame for cluster centers
    centers_df = pd.DataFrame(cluster_centers, columns=feature_names)
    centers_df.index.name = 'Cluster'
    
    print("\nCluster centers (in standardized feature space):")
    print(centers_df)
    
    return cluster_df, centers_df

def visualize_clusters_2d(cluster_df, feature_names):
    """
    Visualize clusters in 2D using the two most important features.
    
    Args:
        cluster_df (pandas.DataFrame): DataFrame with cluster assignments
        feature_names (list): Names of features used for clustering
    """
    print("\n" + "="*50)
    print("VISUALIZING CLUSTERS IN 2D")
    print("="*50)
    
    # Create scatter plots for each pair of features
    # We'll create a few plots with different feature combinations
    
    # Plot 1: First two features
    plt.figure(figsize=(12, 10))
    
    # If there are more than 2 features, create multiple plots
    if len(feature_names) >= 2:
        feature_pairs = [
            (0, 1),  # First two features
            (0, -1), # First and last features
            (-2, -1) # Last two features
        ]
        
        for i, (idx1, idx2) in enumerate(feature_pairs):
            if i >= min(3, len(feature_names)):
                break
                
            feature1 = feature_names[idx1]
            feature2 = feature_names[idx2]
            
            plt.subplot(2, 2, i+1)
            scatter = plt.scatter(cluster_df[feature1], cluster_df[feature2], 
                        c=cluster_df['Cluster'], cmap='viridis', 
                        s=50, alpha=0.8, edgecolors='w')
            plt.xlabel(feature1)
            plt.ylabel(feature2)
            plt.title(f'Clusters: {feature1} vs {feature2}')
            plt.colorbar(scatter, label='Cluster')
            plt.grid(True, alpha=0.3)
    
        # Plot 4: If we have 'Crop' column, show crop types with clusters
        if 'Crop' in cluster_df.columns:
            plt.subplot(2, 2, 4)
            for crop in cluster_df['Crop'].unique():
                crop_data = cluster_df[cluster_df['Crop'] == crop]
                plt.scatter(crop_data[feature_names[0]], crop_data[feature_names[1]], 
                            label=crop, alpha=0.7, s=50, edgecolors='w')
            plt.xlabel(feature_names[0])
            plt.ylabel(feature_names[1])
            plt.title('Clusters by Crop Type')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'clusters_2d.png'))
    plt.close()
    print(f"2D cluster visualization saved to '{os.path.join(IMAGES_DIR, 'clusters_2d.png')}'")

def visualize_clusters_3d(cluster_df, scaled_features):
    """
    Visualize clusters in 3D using PCA for dimensionality reduction if needed.
    
    Args:
        cluster_df (pandas.DataFrame): DataFrame with cluster assignments
        scaled_features (numpy.ndarray): Scaled features used for clustering
    """
    print("\n" + "="*50)
    print("VISUALIZING CLUSTERS IN 3D")
    print("="*50)
    
    # If we have more than 3 features, use PCA to reduce to 3 dimensions
    if scaled_features.shape[1] > 3:
        print("Using PCA to reduce dimensions for 3D visualization")
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(scaled_features)
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.2f}")
        
        # Create a DataFrame with PCA results
        pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2', 'PC3'])
        pca_df['Cluster'] = cluster_df['Cluster'].values
        
        # Create 3D scatter plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each cluster
        for cluster in sorted(pca_df['Cluster'].unique()):
            cluster_data = pca_df[pca_df['Cluster'] == cluster]
            ax.scatter(cluster_data['PC1'], cluster_data['PC2'], cluster_data['PC3'], 
                       label=f'Cluster {cluster}', s=50, alpha=0.7)
        
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        ax.set_title('3D Visualization of Clusters (PCA)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, 'clusters_3d_pca.png'))
        plt.close()
        print(f"3D cluster visualization with PCA saved to '{os.path.join(IMAGES_DIR, 'clusters_3d_pca.png')}'")
    
    # If we have exactly 3 features, use them directly
    elif scaled_features.shape[1] == 3:
        feature_names = cluster_df.select_dtypes(include=['float64', 'int64']).columns[:3].tolist()
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each cluster
        for cluster in sorted(cluster_df['Cluster'].unique()):
            cluster_data = cluster_df[cluster_df['Cluster'] == cluster]
            ax.scatter(cluster_data[feature_names[0]], 
                       cluster_data[feature_names[1]], 
                       cluster_data[feature_names[2]], 
                       label=f'Cluster {cluster}', s=50, alpha=0.7)
        
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_zlabel(feature_names[2])
        ax.set_title('3D Visualization of Clusters')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, 'clusters_3d.png'))
        plt.close()
        print(f"3D cluster visualization saved to '{os.path.join(IMAGES_DIR, 'clusters_3d.png')}'")
    
    # If we have fewer than 3 features, we can't create a 3D plot
    else:
        print("Not enough features for 3D visualization. Skipping 3D plot.")

def identify_cluster_outliers(cluster_df, feature_names):
    """
    Identify outliers within each cluster based on statistical methods.
    
    Args:
        cluster_df (pandas.DataFrame): DataFrame with cluster assignments
        feature_names (list): Names of features used for clustering
        
    Returns:
        pandas.DataFrame: DataFrame with outlier flags
    """
    print("\n" + "="*50)
    print("IDENTIFYING OUTLIERS WITHIN CLUSTERS")
    print("="*50)
    
    # Create a copy of the dataframe to add outlier information
    outlier_df = cluster_df.copy()
    outlier_df['is_outlier'] = False
    
    # For each cluster, identify outliers using the IQR method
    for cluster in sorted(outlier_df['Cluster'].unique()):
        cluster_data = outlier_df[outlier_df['Cluster'] == cluster]
        
        # Calculate outliers for each feature
        for feature in feature_names:
            Q1 = cluster_data[feature].quantile(0.25)
            Q3 = cluster_data[feature].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier boundaries (1.5 * IQR)
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Mark outliers
            feature_outliers = (cluster_data[feature] < lower_bound) | (cluster_data[feature] > upper_bound)
            outlier_indices = cluster_data[feature_outliers].index
            outlier_df.loc[outlier_indices, 'is_outlier'] = True
    
    # Count outliers
    outlier_count = outlier_df['is_outlier'].sum()
    print(f"Total outliers identified: {outlier_count} ({outlier_count/len(outlier_df)*100:.1f}% of data)")
    
    # Summarize outliers by cluster
    print("\nOutliers by cluster:")
    for cluster in sorted(outlier_df['Cluster'].unique()):
        cluster_data = outlier_df[outlier_df['Cluster'] == cluster]
        cluster_outliers = cluster_data['is_outlier'].sum()
        print(f"Cluster {cluster}: {cluster_outliers} outliers ({cluster_outliers/len(cluster_data)*100:.1f}% of cluster)")
    
    # Visualize outliers in 2D
    if len(feature_names) >= 2:
        plt.figure(figsize=(12, 10))
        
        # Plot non-outliers
        non_outliers = outlier_df[~outlier_df['is_outlier']]
        plt.scatter(non_outliers[feature_names[0]], non_outliers[feature_names[1]], 
                    c=non_outliers['Cluster'], cmap='viridis', 
                    s=50, alpha=0.7, edgecolors='w', label='Normal')
        
        # Plot outliers
        outliers = outlier_df[outlier_df['is_outlier']]
        plt.scatter(outliers[feature_names[0]], outliers[feature_names[1]], 
                    c=outliers['Cluster'], cmap='viridis', 
                    s=100, alpha=1.0, edgecolors='red', linewidth=2, marker='X', label='Outlier')
        
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.title('Cluster Outliers')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, 'cluster_outliers.png'))
        plt.close()
        print(f"Cluster outliers visualization saved to '{os.path.join(IMAGES_DIR, 'cluster_outliers.png')}'")
    
    # Analyze outlier characteristics
    if outlier_count > 0:
        print("\nOutlier characteristics:")
        outliers = outlier_df[outlier_df['is_outlier']]
        non_outliers = outlier_df[~outlier_df['is_outlier']]
        
        for feature in feature_names:
            outlier_mean = outliers[feature].mean()
            non_outlier_mean = non_outliers[feature].mean()
            print(f"{feature}: Outlier mean = {outlier_mean:.2f}, Non-outlier mean = {non_outlier_mean:.2f}, Difference = {outlier_mean - non_outlier_mean:.2f}")
    
    return outlier_df

def summarize_clustering_results(cluster_df, centers_df, feature_names):
    """
    Summarize the results of the clustering analysis.
    
    Args:
        cluster_df (pandas.DataFrame): DataFrame with cluster assignments
        centers_df (pandas.DataFrame): DataFrame with cluster centers
        feature_names (list): Names of features used for clustering
    """
    print("\n" + "="*50)
    print("CLUSTERING ANALYSIS SUMMARY")
    print("="*50)
    
    # Number of clusters
    num_clusters = len(centers_df)
    print(f"Number of clusters: {num_clusters}")
    
    # Cluster sizes
    cluster_sizes = cluster_df['Cluster'].value_counts().sort_index()
    print("\nCluster sizes:")
    for cluster, size in cluster_sizes.items():
        print(f"Cluster {cluster}: {size} samples ({size/len(cluster_df)*100:.1f}%)")
    
    # Analyze cluster characteristics
    print("\nCluster characteristics:")
    
    # For each cluster, calculate statistics for key features
    for cluster in sorted(cluster_df['Cluster'].unique()):
        print(f"\nCluster {cluster}:")
        cluster_data = cluster_df[cluster_df['Cluster'] == cluster]
        
        # Calculate mean values for each feature
        for feature in feature_names:
            feature_mean = cluster_data[feature].mean()
            feature_std = cluster_data[feature].std()
            overall_mean = cluster_df[feature].mean()
            diff_from_overall = ((feature_mean - overall_mean) / overall_mean) * 100
            
            # Determine if this feature is significantly higher or lower than average
            if abs(diff_from_overall) > 10:  # More than 10% difference
                direction = "higher" if diff_from_overall > 0 else "lower"
                print(f"  - {feature}: {feature_mean:.2f} (±{feature_std:.2f}), {abs(diff_from_overall):.1f}% {direction} than average")
    
    # If 'Yield' column exists, analyze yield by cluster
    if 'Yield' in cluster_df.columns:
        print("\nYield by cluster:")
        for cluster in sorted(cluster_df['Cluster'].unique()):
            cluster_data = cluster_df[cluster_df['Cluster'] == cluster]
            yield_mean = cluster_data['Yield'].mean()
            yield_std = cluster_data['Yield'].std()
            overall_yield_mean = cluster_df['Yield'].mean()
            diff_from_overall = ((yield_mean - overall_yield_mean) / overall_yield_mean) * 100
            
            direction = "higher" if diff_from_overall > 0 else "lower"
            print(f"Cluster {cluster}: {yield_mean:.2f} (±{yield_std:.2f}), {abs(diff_from_overall):.1f}% {direction} than average")
    
    # If 'Crop' column exists, analyze crop distribution by cluster
    if 'Crop' in cluster_df.columns:
        print("\nCrop distribution by cluster:")
        crop_cluster_counts = pd.crosstab(cluster_df['Cluster'], cluster_df['Crop'], normalize='index') * 100
        print(crop_cluster_counts.round(1))
        
        # Identify dominant crops in each cluster
        print("\nDominant crops by cluster:")
        for cluster in sorted(cluster_df['Cluster'].unique()):
            cluster_crops = crop_cluster_counts.loc[cluster]
            dominant_crop = cluster_crops.idxmax()
            dominant_pct = cluster_crops.max()
            print(f"Cluster {cluster}: {dominant_crop} ({dominant_pct:.1f}%)")
    
    # Create a summary plot
    plt.figure(figsize=(12, 8))
    
    # If we have yield data, plot yield by cluster
    if 'Yield' in cluster_df.columns:
        plt.subplot(2, 1, 1)
        sns.boxplot(x='Cluster', y='Yield', data=cluster_df)
        plt.title('Yield Distribution by Cluster')
        plt.grid(True, alpha=0.3)
    
    # Plot cluster sizes
    plt.subplot(2, 1, 2)
    cluster_sizes.plot(kind='bar')
    plt.title('Cluster Sizes')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Samples')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'clustering_summary.png'))
    plt.close()
    print(f"\nClustering summary plot saved to '{os.path.join(IMAGES_DIR, 'clustering_summary.png')}'")
    
    print("\nClustering analysis complete. Use these insights to understand patterns in crop productivity.")

def perform_clustering_analysis(df):
    """
    Perform clustering analysis on the dataset to identify patterns in crop productivity.
    
    Args:
        df (pandas.DataFrame): Dataset to analyze
        
    Returns:
        pandas.DataFrame: DataFrame with cluster assignments and outlier flags
    """
    print("\n" + "="*50)
    print("CLUSTERING ANALYSIS")
    print("="*50)
    
    # Step 1: Preprocess data for clustering
    cluster_df, scaled_features, feature_names, scaler = preprocess_data_for_clustering(df)
    
    # Step 2: Determine the optimal number of clusters
    optimal_k = determine_optimal_clusters(scaled_features)
    
    # Step 3: Perform K-means clustering with the optimal number of clusters
    cluster_df, centers_df = perform_kmeans_clustering(cluster_df, scaled_features, optimal_k, feature_names)
    
    # Step 4: Visualize clusters in 2D
    visualize_clusters_2d(cluster_df, feature_names)
    
    # Step 5: Visualize clusters in 3D (if possible)
    visualize_clusters_3d(cluster_df, scaled_features)
    
    # Step 6: Identify outliers within clusters
    outlier_df = identify_cluster_outliers(cluster_df, feature_names)
    
    # Step 7: Summarize clustering results
    summarize_clustering_results(outlier_df, centers_df, feature_names)
    
    return outlier_df

def prepare_data_for_modeling(df, target_column='Yield', test_size=0.2, random_state=42):
    """
    Prepare data for supervised learning models.
    
    Args:
        df (pandas.DataFrame): Dataset to prepare
        target_column (str): Name of the target column
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names, scaler)
    """
    print("\n" + "="*50)
    print("PREPARING DATA FOR PREDICTIVE MODELING")
    print("="*50)
    
    # Make a copy of the dataframe to avoid modifying the original
    model_df = df.copy()
    
    # If we have cluster columns from previous analysis, remove them
    cols_to_drop = [col for col in model_df.columns if col in ['Cluster', 'is_outlier']]
    if cols_to_drop:
        model_df = model_df.drop(columns=cols_to_drop)
        print(f"Removed columns from clustering analysis: {cols_to_drop}")
    
    # Separate features and target
    X = model_df.drop(columns=[target_column])
    y = model_df[target_column]
    
    # Get feature names for later use
    feature_names = X.columns.tolist()
    print(f"Features for modeling: {feature_names}")
    print(f"Target variable: {target_column}")
    
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    # Handle categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if categorical_cols:
        print(f"Encoding categorical features: {categorical_cols}")
        # Use one-hot encoding for categorical features
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        
        # Apply to both train and test sets
        X_train_cat = encoder.fit_transform(X_train[categorical_cols])
        X_test_cat = encoder.transform(X_test[categorical_cols])
        
        # Get encoded feature names
        encoded_feature_names = []
        for i, category in enumerate(encoder.categories_):
            for j, cat_value in enumerate(category[1:], 1):  # Skip the first category (dropped)
                encoded_feature_names.append(f"{categorical_cols[i]}_{cat_value}")
        
        # Scale numerical features
        if numerical_cols:
            scaler = StandardScaler()
            X_train_num = scaler.fit_transform(X_train[numerical_cols])
            X_test_num = scaler.transform(X_test[numerical_cols])
            
            # Combine numerical and categorical features
            X_train_scaled = np.hstack([X_train_num, X_train_cat])
            X_test_scaled = np.hstack([X_test_num, X_test_cat])
            
            # Update feature names
            feature_names = numerical_cols + encoded_feature_names
        else:
            X_train_scaled = X_train_cat
            X_test_scaled = X_test_cat
            feature_names = encoded_feature_names
    else:
        # If no categorical features, just scale numerical features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
    print("Data preprocessing complete")
    print(f"Final feature set size: {len(feature_names)} features")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a regression model using multiple metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target values
        model_name (str): Name of the model for display
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Print metrics
    print(f"\n{model_name} Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    
    # Create a dictionary of metrics for comparison
    metrics = {
        'Model': model_name,
        'R²': r2,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse
    }
    
    # Visualize actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Yield')
    plt.ylabel('Predicted Yield')
    plt.title(f'{model_name}: Actual vs Predicted Yield')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(IMAGES_DIR, f'{model_name.lower().replace(" ", "_")}_predictions.png'))
    plt.close()
    
    return metrics

def plot_feature_importance(model, feature_names, model_name):
    """
    Plot feature importance for models.
    
    Args:
        model: Trained model (tree-based or linear regression)
        feature_names (list): Names of features
        model_name (str): Name of the model for display
    """
    # For tree-based models with feature_importances_ attribute
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importances
        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Importance - {model_name}')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, f'{model_name.lower().replace(" ", "_")}_feature_importance.png'))
        plt.close()
        print(f"\nFeature importance plot saved for {model_name}")
    
    # For linear regression models with coef_ attribute
    elif hasattr(model, 'coef_'):
        # Get coefficients
        coefficients = model.coef_
        
        # Create a DataFrame for better visualization
        coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
        
        # Sort by absolute coefficient values
        coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
        coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
        
        # Plot coefficients
        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Coefficients - {model_name}')
        bars = plt.barh(range(len(coef_df)), coef_df['Coefficient'], align='center')
        
        # Color bars based on coefficient sign
        for i, bar in enumerate(bars):
            if coef_df['Coefficient'].iloc[i] < 0:
                bar.set_color('salmon')
            else:
                bar.set_color('skyblue')
                
        plt.yticks(range(len(coef_df)), coef_df['Feature'])
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.xlabel('Coefficient Value')
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, f'{model_name.lower().replace(" ", "_")}_coefficients.png'))
        plt.close()
        print(f"\nCoefficient plot saved for {model_name}")
    
    else:
        print(f"\nFeature importance not available for {model_name}")

def train_linear_regression(X_train, y_train, X_test, y_test, feature_names):
    """
    Train and evaluate a Linear Regression model.
    
    Args:
        X_train: Training features
        y_train: Training target values
        X_test: Test features
        y_test: Test target values
        feature_names (list): Names of features
        
    Returns:
        tuple: (model, metrics)
    """
    print("\n" + "="*50)
    print("LINEAR REGRESSION MODEL")
    print("="*50)
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Print model coefficients
    print("Model coefficients:")
    for feature, coef in zip(feature_names, model.coef_):
        print(f"{feature}: {coef:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    
    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test, "Linear Regression")
    
    return model, metrics

def train_decision_tree(X_train, y_train, X_test, y_test, feature_names):
    """
    Train and evaluate a Decision Tree Regressor model.
    
    Args:
        X_train: Training features
        y_train: Training target values
        X_test: Test features
        y_test: Test target values
        feature_names (list): Names of features
        
    Returns:
        tuple: (model, metrics)
    """
    print("\n" + "="*50)
    print("DECISION TREE REGRESSOR MODEL")
    print("="*50)
    
    # Create and train the model
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test, "Decision Tree")
    
    # Plot feature importance
    plot_feature_importance(model, feature_names, "Decision Tree")
    
    return model, metrics

def train_random_forest(X_train, y_train, X_test, y_test, feature_names):
    """
    Train and evaluate a Random Forest Regressor model.
    
    Args:
        X_train: Training features
        y_train: Training target values
        X_test: Test features
        y_test: Test target values
        feature_names (list): Names of features
        
    Returns:
        tuple: (model, metrics)
    """
    print("\n" + "="*50)
    print("RANDOM FOREST REGRESSOR MODEL")
    print("="*50)
    
    # Create and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test, "Random Forest")
    
    # Plot feature importance
    plot_feature_importance(model, feature_names, "Random Forest")
    
    return model, metrics

def train_gradient_boosting(X_train, y_train, X_test, y_test, feature_names):
    """
    Train and evaluate a Gradient Boosting Regressor model (XGBoost).
    
    Args:
        X_train: Training features
        y_train: Training target values
        X_test: Test features
        y_test: Test target values
        feature_names (list): Names of features
        
    Returns:
        tuple: (model, metrics)
    """
    print("\n" + "="*50)
    print("GRADIENT BOOSTING REGRESSOR MODEL (XGBOOST)")
    print("="*50)
    
    # Create and train the model
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test, "XGBoost")
    
    # Plot feature importance
    plot_feature_importance(model, feature_names, "XGBoost")
    
    return model, metrics

def train_neural_network(X_train, y_train, X_test, y_test, feature_names):
    """
    Train and evaluate a Neural Network Regressor model (MLP).
    
    Args:
        X_train: Training features
        y_train: Training target values
        X_test: Test features
        y_test: Test target values
        feature_names (list): Names of features
        
    Returns:
        tuple: (model, metrics)
    """
    print("\n" + "="*50)
    print("NEURAL NETWORK REGRESSOR MODEL (MLP)")
    print("="*50)
    
    # Create and train the model
    model = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test, "Neural Network")
    
    return model, metrics

def compare_models(all_metrics):
    """
    Compare all models and identify the best one.
    
    Args:
        all_metrics (list): List of dictionaries with model metrics
    """
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    
    # Create a DataFrame for comparison
    comparison_df = pd.DataFrame(all_metrics)
    
    # Set 'Model' as the index
    comparison_df.set_index('Model', inplace=True)
    
    # Display the comparison table
    print("\nModel Performance Comparison:")
    print(comparison_df)
    
    # Identify the best model based on R² score
    best_r2_model = comparison_df['R²'].idxmax()
    best_r2_value = comparison_df.loc[best_r2_model, 'R²']
    
    # Identify the best model based on RMSE
    best_rmse_model = comparison_df['RMSE'].idxmin()
    best_rmse_value = comparison_df.loc[best_rmse_model, 'RMSE']
    
    print(f"\nBest model based on R² score: {best_r2_model} (R² = {best_r2_value:.4f})")
    print(f"Best model based on RMSE: {best_rmse_model} (RMSE = {best_rmse_value:.4f})")
    
    # Create a bar chart for R² comparison
    plt.figure(figsize=(12, 6))
    
    # Plot R² scores
    plt.subplot(1, 2, 1)
    comparison_df['R²'].sort_values().plot(kind='barh', color='skyblue')
    plt.title('Model Comparison - R² Score')
    plt.xlabel('R² Score (higher is better)')
    plt.grid(True, alpha=0.3)
    
    # Plot RMSE scores
    plt.subplot(1, 2, 2)
    comparison_df['RMSE'].sort_values(ascending=False).plot(kind='barh', color='salmon')
    plt.title('Model Comparison - RMSE')
    plt.xlabel('RMSE (lower is better)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'model_comparison.png'))
    plt.close()
    print(f"\nModel comparison chart saved to '{os.path.join(IMAGES_DIR, 'model_comparison.png')}'")
    
    return comparison_df, best_r2_model

def perform_predictive_modeling(df):
    """
    Perform predictive modeling to forecast crop yields.
    
    Args:
        df (pandas.DataFrame): Dataset to analyze
        
    Returns:
        tuple: (best_model, comparison_df)
    """
    print("\n" + "="*50)
    print("PREDICTIVE MODELING FOR CROP YIELD")
    print("="*50)
    
    # Step 1: Prepare data for modeling
    X_train, X_test, y_train, y_test, feature_names, scaler = prepare_data_for_modeling(df)
    
    # Step 2: Train and evaluate different models
    all_metrics = []
    
    # Linear Regression
    _, lr_metrics = train_linear_regression(X_train, y_train, X_test, y_test, feature_names)
    all_metrics.append(lr_metrics)
    
    # Decision Tree
    _, dt_metrics = train_decision_tree(X_train, y_train, X_test, y_test, feature_names)
    all_metrics.append(dt_metrics)
    
    # Random Forest
    _, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test, feature_names)
    all_metrics.append(rf_metrics)
    
    # Gradient Boosting (XGBoost)
    _, xgb_metrics = train_gradient_boosting(X_train, y_train, X_test, y_test, feature_names)
    all_metrics.append(xgb_metrics)
    
    # Neural Network
    _, nn_metrics = train_neural_network(X_train, y_train, X_test, y_test, feature_names)
    all_metrics.append(nn_metrics)
    
    # Step 3: Compare all models
    comparison_df, best_model = compare_models(all_metrics)
    
    print(f"\nPredictive modeling complete. The best model for crop yield prediction is {best_model}.")
    
    return best_model, comparison_df

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
    
    # Predictive Modeling
    print("\n5. Predictive Modeling Results:")
    print("   - Five different machine learning models were implemented to predict crop yield.")
    print("   - Models evaluated: Linear Regression, Decision Tree, Random Forest, XGBoost, and Neural Network.")
    print("   - Each model was evaluated using R², MAE, MSE, and RMSE metrics.")
    print("   - The best performing model based on R² score was identified.")
    print("   - Feature importance analysis revealed the most influential factors affecting crop yield.")
    
    # Next Steps
    print("\n6. Next Steps:")
    print("   - Further tune hyperparameters of the best performing model to improve prediction accuracy.")
    print("   - Explore additional features or feature engineering techniques to enhance model performance.")
    print("   - Consider ensemble methods combining multiple models for more robust predictions.")
    print("   - Develop a deployment strategy for the model to assist farmers in yield prediction.")

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
    
    # Perform clustering analysis to identify patterns in crop productivity
    clustered_df = perform_clustering_analysis(df)
    
    # Perform predictive modeling to forecast crop yields
    best_model, model_comparison = perform_predictive_modeling(df)
    
    # Summarize findings
    summarize_findings(df, missing_info)
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    print(f"All visualizations have been saved to the '{IMAGES_DIR}' directory.")
    print(f"Best model for crop yield prediction: {best_model}")

if __name__ == "__main__":
    main()
