import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def desc_length(df, col_name):
    """
    Function for length of str column
    :param df: pd.DataFrame as input
    :param col_name: name of column for which length should be calculated
    :return:
    """

    df.loc[:, col_name + '_length'] = df[col_name].str.len()
    df.loc[df.loc[:, col_name + '_length'].isna(), col_name + '_length'] = 0

    return df


class DescTFIDF:

    def __init__(self):
        self.model = TfidfVectorizer(min_df=2,  max_features=None,
                                     strip_accents='unicode', analyzer='word', token_pattern=r'(?u)\b\w+\b',
                                     ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1)
        self.svd = TruncatedSVD(n_components=25)

    def fit(self, df, col_name):
        X = df.loc[:, col_name].fillna("none").values
        self.model.fit(list(X))
        X_trans = self.model.transform(X)
        self.svd.fit(X_trans)
        X_trans = self.svd.transform(X_trans)
        return pd.DataFrame(X_trans, columns=["desc_tfidf_" + str(i) for i in np.arange(25)])

    def predict(self, df, col_name):
        X = df.loc[:, col_name].fillna("none").values
        X_trans = self.model.transform(X)
        X_trans = self.svd.transform(X_trans)
        return pd.DataFrame(X_trans, columns=["desc_tfidf_" + str(i) for i in np.arange(25)])







