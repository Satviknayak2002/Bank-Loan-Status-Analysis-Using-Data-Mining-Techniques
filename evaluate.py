from sklearn.metrics import classification_report

def evaluate_models(models, X_test, y_test):
    for name, model in models.items():
        print(f"\n{name} Evaluation:\n")
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
