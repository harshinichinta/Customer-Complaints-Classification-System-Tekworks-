import pandas as pd
from collections import Counter

p = r'c:\Project(Tekworks)\data\complaints.csv'
print('Loading:', p)
try:
    df = pd.read_csv(p)
except Exception as e:
    print('ERROR loading CSV:', e)
    raise

print('\n--- Basic Info ---')
rows, cols = df.shape
print('Number of rows:', rows)
print('Number of columns:', cols)
print('Column names:', list(df.columns))
print('\n--- Data types ---')
print(df.dtypes)
print('\n--- Missing values (per column) ---')
print(df.isnull().sum())
print('\n--- Duplicate records ---')
print('Duplicate rows count:', df.duplicated().sum())

print('\n--- Unique values (per column) ---')
for c in df.columns:
    try:
        nunique = df[c].nunique(dropna=False)
    except Exception:
        nunique = 'ERR'
    print(f"{c}: {nunique} unique")

text_cols = []
num_cols = []
cat_cols = []
for c in df.columns:
    if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c]):
        text_cols.append(c)
    elif pd.api.types.is_numeric_dtype(df[c]):
        num_cols.append(c)
    else:
        cat_cols.append(c)

print('\nText columns:', text_cols)
print('Numerical columns:', num_cols)
print('Other categorical columns (detected):', cat_cols)

# Determine candidate target column: pick categorical with small unique values but not text narrative
candidates = []
for c in df.columns:
    if c in text_cols:
        nunique = df[c].nunique(dropna=False)
        mean_len = df[c].astype(str).str.len().mean()
        if mean_len > 50:
            # treat as free text
            continue
        # otherwise candidate
        candidates.append((c, nunique))
    elif c in num_cols:
        # numeric could be target but skip
        continue

print('\nCandidate target columns (detected):', candidates)

# Heuristic: choose 'product' if present
target = None
if 'product' in df.columns:
    target = 'product'
elif candidates:
    target = candidates[0][0]

print('\nDetermined target column:', target)

if target is not None:
    print('\n--- Target class distribution ---')
    vc = df[target].value_counts(dropna=False)
    print(vc)
    print('\nNumber of target classes:', vc.shape[0])

print('\n--- Sample rows (5) ---')
print(df.head(5).to_string(index=False))

print('\n--- Narrative length stats (if narrative present) ---')
if 'narrative' in df.columns:
    s = df['narrative'].astype(str).str.len()
    print(s.describe())

print('\nDone')
