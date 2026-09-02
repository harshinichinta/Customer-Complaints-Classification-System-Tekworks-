import re
import pandas as pd


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.lower()
    text = re.sub(r"\r|\n", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure expected columns exist
    if 'narrative' in df.columns:
        df['text'] = df['narrative'].astype(str).apply(clean_text)
    else:
        # If there's only one text column, try to find it
        text_cols = [c for c in df.columns if df[c].dtype == object]
        if text_cols:
            df['text'] = df[text_cols[0]].astype(str).apply(clean_text)
        else:
            raise ValueError('No text column found')
    return df
