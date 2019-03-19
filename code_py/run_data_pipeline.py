import pandas as pd

from code_py.load_data import colnames_to_lower, load_jsons, load_meta
from code_py.feature_engineering import desc_length, DescTFIDF


# Import tabular data
df_tab_train = pd.read_csv("data/raw/train.csv")
df_tab_test = pd.read_csv("data/raw/test.csv")
df_breed_labels = pd.read_csv("data/raw/breed_labels.csv")
df_color_labels = pd.read_csv("data/raw/color_labels.csv")
df_state_labels = pd.read_csv("data/raw/state_labels.csv")

# Fix column names
df_tab_train = colnames_to_lower(df_tab_train)
df_tab_test = colnames_to_lower(df_tab_test)
df_breed_labels = colnames_to_lower(df_breed_labels)
df_color_labels = colnames_to_lower(df_color_labels)
df_state_labels = colnames_to_lower(df_state_labels)

# Extract ids
train_id = df_tab_train['petid']
test_id = df_tab_test['petid']

# Load JSONs
df_sentiment_train = load_jsons(train_id, 'train')
df_sentiment_test = load_jsons(test_id, 'test')

# Load sentiments
df_meta_train = load_meta(train_id, 'train')
df_meta_test = load_meta(test_id, 'test')

# Calculate string length of description
df_tab_train = desc_length(df_tab_train, 'description')
df_tab_test = desc_length(df_tab_test, 'description')

# Calculate TFIDF
mod = DescTFIDF()
df_description_tfidf_train = mod.fit(df_tab_train, 'description')
df_description_tfidf_test = mod.predict(df_tab_test, 'description')
df_tab_train = pd.concat([df_tab_train, df_description_tfidf_train], axis=1)
df_tab_test = pd.concat([df_tab_test, df_description_tfidf_test], axis=1)

# Merge
df_train = pd.merge(df_tab_train, df_sentiment_train, on="petid")
df_test = pd.merge(df_tab_test, df_sentiment_test, on="petid")

# Export to csv
df_train.to_csv("data/prepared/train.csv")
df_test.to_csv("data/prepared/test.csv")

# Export to pickle
df_train.to_pickle("data/prepared/train.pkl")
df_test.to_pickle("data/prepared/test.pkl")
