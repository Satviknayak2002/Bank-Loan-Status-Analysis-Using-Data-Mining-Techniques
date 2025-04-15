from train import train_models
from evaluate import evaluate_models
from utils.preprocess import load_and_preprocess

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess("data/dataset.csv")
    models = train_models(X_train, y_train)
    evaluate_models(models, X_test, y_test)
