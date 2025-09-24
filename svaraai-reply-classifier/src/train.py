import argparse
import joblib
from sklearn.metrics import accuracy_score, f1_score
from src.data_processing import load_data
from src.features import build_baseline_pipeline

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    return acc, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./reply_classification_dataset.csv')
    parser.add_argument('--out', type=str, default='artifacts/baseline_model.pkl')
    args = parser.parse_args()

    train, test = load_data(args.data_path)
    X_train, y_train = train['text'], train['label']
    X_test, y_test = test['text'], test['label']

    pipe = build_baseline_pipeline()
    pipe.fit(X_train, y_train)
    acc, f1 = evaluate_model(pipe, X_test, y_test)
    print(f"Baseline -- acc: {acc:.4f}, f1: {f1:.4f}")

    joblib.dump(pipe, args.out)
    print(f"Saved baseline model to {args.out}")

if __name__ == '__main__':
    main()
