import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# 1. Impute Missing Values
def impute_missing_values(data, strategy='mean'):
    """
    Fills missing values in a dataset based on the given strategy.
    Supports mean, median, and mode.
    """
    for col in data.columns:
        if data[col].dtype in ['int64', 'float64']:
            if strategy == 'mean':
                fill_value = data[col].mean()
            elif strategy == 'median':
                fill_value = data[col].median()
            elif strategy == 'mode':
                fill_value = data[col].mode()[0]
            else:
                continue
        else:
            fill_value = data[col].mode()[0]
        
        data[col] = data[col].fillna(fill_value)  # SAFE: Avoids chained assignment warning
    return data


# 2. Remove Duplicate Rows
def remove_duplicates(data):
    """
    Removes exact duplicate rows from the dataset.
    """
    return data.drop_duplicates()


# 3. Normalize Numerical Data
def normalize_data(data, method='minmax'):
    """
    Scales numeric features using MinMaxScaler or StandardScaler.
    """
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid normalization method. Use 'minmax' or 'standard'.")

    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    return data


# 4. Remove Redundant Features Based on Correlation
def remove_redundant_features(data, threshold=0.9):
    """
    Drops highly correlated features to reduce multicollinearity.
    """
    corr_matrix = data.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > threshold)]
    return data.drop(columns=to_drop)

def simple_model(input_data, split_data=True, scale_data=False, print_report=False):
    """
    Trains and evaluates a logistic regression model.
    Assumes the first column is the target.
    Automatically filters target to binary classes and ensures compatibility.
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, accuracy_score

    print("\n Starting model pipeline...")

    # Step 1: Clean the target column
    target_col = input_data.columns[0]

    # Drop rows with missing target
    input_data = input_data.dropna(subset=[target_col])

    # Extract target and features
    target = pd.to_numeric(input_data[target_col], errors='coerce')  # handles float, string, NaN
    features = input_data.iloc[:, 1:]

    # Keep only 0 and 1
    valid_idx = target[target.isin([0, 1])].index
    target = target.loc[valid_idx].astype(int)
    features = features.loc[valid_idx]

    # Encode features
    features = pd.get_dummies(features)

    print("\n Sanity Check on Target:")
    print("Target dtype:", target.dtype)
    print("Target values:\n", target.value_counts())

    # Train/test split
    if split_data:
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, stratify=target, random_state=42
        )
    else:
        X_train = X_test = features
        y_train = y_test = target

    # Optional scaling
    if scale_data:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Model training
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n Model Accuracy: {acc:.4f}")

    if print_report:
        print("\n Classification Report:")
        print(classification_report(y_test, y_pred))
