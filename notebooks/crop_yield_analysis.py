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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

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
    
    # Perform clustering analysis to identify patterns in crop productivity
    clustered_df = perform_clustering_analysis(df)
    
    # Summarize findings
    summarize_findings(df, missing_info)
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    print(f"All visualizations have been saved to the '{IMAGES_DIR}' directory.")

if __name__ == "__main__":
    main()
