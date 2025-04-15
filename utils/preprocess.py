import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)

    # Encode categorical features if any
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    X = df.drop('target', axis=1)  # Replace 'target' with actual label column
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)
