# LDA_analysis

This repository contains Python code for performing Linear Discriminant Analysis (LDA) on a dataset. LDA is a dimensionality reduction technique commonly used in machine learning and statistics for feature extraction and data visualization.

## Overview

Linear Discriminant Analysis (LDA) is a method used to find the linear combinations of features that best separate two or more classes of data. It is commonly used for dimensionality reduction and classification in machine learning tasks.

The main script `LDA_analysis.py` includes functions to perform LDA and visualize the results. It utilizes the `LinearDiscriminantAnalysis` class from the scikit-learn library for LDA computation.

## How It Works

The script `LDA_analysis.py` works as follows:

- It reads a cleaned dataset from the file `data_clean.csv`.
- It defines all possible combinations of two parameters: standardization and outlier inclusion.
- For each combination, it computes the explained variance ratio using LDA and prints the results.
- It also generates visualizations of LDA results for each combination, showing the separation of classes in two-dimensional space.

## Dataset Information

The dataset used in this analysis contains the following columns:

- fixed.acidity
- volatile.acidity
- citric.acid
- residual.sugar
- chlorides
- free.sulfur.dioxide
- total.sulfur.dioxide
- density
- pH
- sulphates
- alcohol
- quality
- outlier

Each row represents a sample of wine, and the columns contain various chemical properties and quality metrics. The `quality` column represents the quality rating of the wine, and the `outlier` column indicates whether the sample is an outlier.

The script iterates over all possible combinations of two parameters: standardization and outlier inclusion. Each combination is printed along with the corresponding explained variance ratio. Visualizations are also generated for each combination.
