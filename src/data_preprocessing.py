import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_clean_data(path):
    df = pd.read_csv(path)

    # Convert target to binary
    df["target"] = df["num"].apply(lambda x: 1 if x > 0 else 0)

    # Drop unnecessary columns
    df = df.drop(columns=["id", "num"])

    # One-hot encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    # 🔥 HANDLE MISSING VALUES
    df = df.fillna(df.mean())

    return df


def split_data(df):
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled