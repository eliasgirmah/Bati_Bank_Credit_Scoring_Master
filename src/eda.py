import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Skimpy for column cleaning
from skimpy import clean_columns

warnings.filterwarnings("ignore")

# Set global plotting params
plt.rcParams["figure.figsize"] = (10, 6)
sns.set_style("whitegrid")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)

def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Data Shape: {df.shape}")
    print(f"Data Types:\n{df.dtypes}")
    return df

def summary_statistics(df):
    summary_stats = df.describe()
    print(f"Summary Statistics:\n{summary_stats}")
    return summary_stats

def visualize_numerical_features(df, output_dir='plots'):
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(12, 10))
    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(3, 3, i)
        sns.histplot(df[feature], bins=30, kde=True)
        plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    save_plot(output_dir, 'numerical_features_distribution.png')

def visualize_categorical_features(df, output_dir='plots'):
    categorical_features = df.select_dtypes(include=['object', 'bool']).columns
    sample_size = min(10000, len(df))  
    df_sample = df.sample(sample_size)
    
    for i in range(0, len(categorical_features), 10):
        batch_features = categorical_features[i:i+10]
        num_rows = (len(batch_features) + 2) // 3
        fig, axes = plt.subplots(num_rows, 3, figsize=(12, 5 * num_rows))
        axes = axes.flatten()

        for j, feature in enumerate(batch_features):
            sns.countplot(data=df_sample, x=feature, ax=axes[j])
            axes[j].set_title(f'Distribution of {feature}')
            axes[j].tick_params(axis='x', rotation=45)

        for k in range(j + 1, num_rows * 3):
            fig.delaxes(axes[k])

        plt.tight_layout()
        save_plot(output_dir, f'categorical_distribution_batch_{i//10 + 1}.png')

def correlation_analysis(df):
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = df[numerical_features].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def identify_missing_values(df):
    missing_values = df.isnull().sum()
    print("Missing Values:\n", missing_values[missing_values > 0])
    return missing_values[missing_values > 0]

def detect_outliers(df):
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(12, 10))
    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(3, 3, i)
        sns.boxplot(x=df[feature])
        plt.title(f'Box Plot of {feature}')
    plt.tight_layout()
    plt.show()

def feature_engineering(df):
    agg_features = df.groupby('CustomerId').agg(
        Total_Transaction_Amount=('log_amount', 'sum'),
        Average_Transaction_Amount=('log_amount', 'mean'),
        Transaction_Count=('log_amount', 'count'),
        Std_Transaction_Amount=('log_amount', 'std')
    ).reset_index()
    df = df.merge(agg_features, on='CustomerId', how='left')
    return df

def save_plot(output_dir, filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# Example usage of functions
if __name__ == '__main__':
    df = load_data("./data/cleaned_data.csv")
    summary_statistics(df)
    visualize_numerical_features(df)
    visualize_categorical_features(df)
    correlation_analysis(df)
    identify_missing_values(df)
    detect_outliers(df)
    df = feature_engineering(df)
