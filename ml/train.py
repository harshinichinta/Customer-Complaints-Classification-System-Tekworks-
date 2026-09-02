import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from preprocessing import load_and_clean


def train(output_dir='models', data_path='data/complaints.csv'):
    os.makedirs(output_dir, exist_ok=True)
    df = load_and_clean(data_path)
    if 'product' not in df.columns:
        raise ValueError('Expected target column "product"')
    df = df[["text", "product"]].dropna()

    X = df['text'].astype(str).to_numpy()
    y = df['product'].astype(str).to_numpy()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    X_train_t = vectorizer.fit_transform(X_train)
    X_test_t = vectorizer.transform(X_test)

    models = {
        'LogisticRegression': LogisticRegression(max_iter=2000, solver='saga', n_jobs=-1),
        'MultinomialNB': MultinomialNB(),
        'LinearSVC': LinearSVC()
    }

    results = {}
    trained_models = {}
    for name, model in models.items():
        print('Training', name)
        model.fit(X_train_t, y_train)
        y_pred = model.predict(X_test_t)
        acc = accuracy_score(y_test, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()
        results[name] = {'accuracy': acc, 'precision': p, 'recall': r, 'f1_weighted': f1, 'confusion_matrix': cm}
        trained_models[name] = model

    # Select best by weighted f1
    best_name = max(results.keys(), key=lambda k: results[k]['f1_weighted'])
    best_model = trained_models[best_name]

    # Save artifacts
    joblib.dump(best_model, os.path.join(output_dir, 'final_model.joblib'))
    joblib.dump(vectorizer, os.path.join(output_dir, 'vectorizer.joblib'))
    joblib.dump(le, os.path.join(output_dir, 'label_encoder.joblib'))

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump({'results': results, 'best_model': best_name}, f, indent=2)

    print('Training complete. Best model:', best_name)
    return results, best_name


if __name__ == '__main__':
    r, best = train()
    print('Saved model:', best)
