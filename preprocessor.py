from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
import numpy as np

class preprocess(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.num_features = make_column_selector(dtype_include=np.number)
        self.cat_features = make_column_selector(dtype_exclude=np.number)
        
        self.num_pipeline = make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler()
        )
        self.cat_pipeline = make_pipeline(
            SimpleImputer(strategy='most_frequent'),
            OneHotEncoder()
        ) 
        
        self.preprocessor = ColumnTransformer([
            ('num', self.num_pipeline, self.num_features),
            ('cat', self.cat_pipeline, self.cat_features)
        ])
        
    def fit(self, X, y=None):
        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        return self.preprocessor.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.preprocessor.get_feature_names_out(input_features)
