"""Data loading, splitting, and preprocessing."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from config import CSV_PATH


def load_data():
    """Load the merged NHANES CSV and return train/test DataFrames."""
    df = pd.read_csv(CSV_PATH)
    train_df = df[df['SPLIT'] == 'train'].copy()
    test_df = df[df['SPLIT'] == 'test'].copy()
    return train_df, test_df


def prepare_features(train_df, test_df, feature_cols):
    """Impute missing values (median) and standardize features.

    Returns scaled numpy arrays and the fitted imputer/scaler
    for reproducibility.
    """
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_train = scaler.fit_transform(imputer.fit_transform(train_df[feature_cols]))
    X_test = scaler.transform(imputer.transform(test_df[feature_cols]))
    y_train = train_df['STROKE'].values
    y_test = test_df['STROKE'].values

    return X_train, X_test, y_train, y_test, imputer, scaler


def get_missing_rates(df, feature_cols):
    """Return Series of missing-data percentages for the given columns."""
    return df[feature_cols].isnull().mean() * 100
