import pandas as pd
import re
from sklearn.model_selection import train_test_split

DATA_PATH = "./reply_classification_dataset.csv"

LABEL_MAP = {"positive": 0, "negative": 1, "neutral": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"https?://\S+", "", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_data(path: str = DATA_PATH, test_size: float = 0.2, random_state: int = 42):
    df = pd.read_csv(path)
    if 'text' not in df.columns:
        text_col = [c for c in df.columns if 'text' in c.lower()]
        if text_col:
            df = df.rename(columns={text_col[0]: 'text'})
    if 'label' not in df.columns:
        raise ValueError('Dataset must contain a "text" and "label" column')
    df['text'] = df['text'].apply(clean_text)
    df = df.dropna(subset=['text'])
    df['label'] = df['label'].str.lower().map(LABEL_MAP)
    df = df.dropna(subset=['label'])
    train, test = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=random_state)
    return train.reset_index(drop=True), test.reset_index(drop=True)
