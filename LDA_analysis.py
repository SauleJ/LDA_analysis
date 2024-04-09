import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler
from itertools import product

def plot_lda(data, standardized=True, include_outliers=False):

    # Extract features and target variable
    features = data.drop(columns=['outlier', 'quality'])
    quality = data['quality']
    outliers = data['outlier']

    # Standardize features if necessary
    if standardized:
        scaler = MinMaxScaler()
        features = scaler.fit_transform(features)

    # Remove outliers if necessary
    if not include_outliers:
        features = features[outliers == 0]
        quality = quality[outliers == 0]
        outliers = outliers[outliers == 0]

    # Apply LDA model
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda_results = lda.fit_transform(features, quality)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(lda_results[outliers == 0, 0], lda_results[outliers == 0, 1], c=quality[outliers == 0], cmap='viridis', alpha=0.5)
    plt.scatter(lda_results[outliers == 1, 0], lda_results[outliers == 1, 1], c='red', marker='x', label='Outliers', alpha=0.7)
    
    # Title based on parameters
    title = 'LDA: '
    if standardized and include_outliers:
        title += "Standardized with Outliers"
    elif standardized:
        title += "Standardized without Outliers"
    elif include_outliers:
        title += "Not Standardized with Outliers"
    else:
        title += "Not Standardized without Outliers"
    
    plt.title(title)
    plt.xlabel('Linear Discriminant 1')
    plt.ylabel('Linear Discriminant 2')
    plt.colorbar(label='Wine Quality')
    plt.legend(loc='upper right')
    plt.show()

def lda_explained_variance(data, standardize=True, include_outliers=True):

    X = data.drop(columns=['outlier', 'quality'])
    y = data['quality']

    # Exclude outliers if necessary
    if not include_outliers:
        outliers = data['outlier'] == 1
        X = X[~outliers]
        y = y[~outliers]

    # Standardize features if necessary
    if standardize:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    # Apply LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    explained_variance = lda.explained_variance_ratio_

    # Create DataFrame
    df = pd.DataFrame({
        'LD': range(1, len(explained_variance) + 1),
        'Explained Variance Ratio': explained_variance
    })

    return df

# Read data from CSV
data = pd.read_csv('data_clean.csv')

# Define possible parameter combinations
parameter_combinations = product([True, False], repeat=2)

# Iterate over all combinations and print results
for standardized, include_outliers in parameter_combinations:
    print(f"Standardized: {standardized}, Include Outliers: {include_outliers}")
    print(lda_explained_variance(data, standardize=standardized, include_outliers=include_outliers))
    plot_lda(data, standardized=standardized, include_outliers=include_outliers)
