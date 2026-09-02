import joblib
from sklearn.preprocessing import LabelEncoder


def save_artifacts(path: str, model, vectorizer, label_encoder: LabelEncoder):
    joblib.dump(model, path + '/final_model.joblib')
    joblib.dump(vectorizer, path + '/vectorizer.joblib')
    joblib.dump(label_encoder, path + '/label_encoder.joblib')


def load_artifacts(path: str):
    model = joblib.load(path + '/final_model.joblib')
    vectorizer = joblib.load(path + '/vectorizer.joblib')
    label_encoder = joblib.load(path + '/label_encoder.joblib')
    return model, vectorizer, label_encoder
