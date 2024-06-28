from preprocessor import preprocess
from trainer import ModelTrainer
import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import randint
from scipy.stats import uniform
# Load the data
df = pd.read_csv(r'Data\BikeDataset.csv')
print("Data loaded successfully")

print(df.head())

print("-"*150)

print(f"Shape of the DataFrame: {df.shape}")

print("-"*150)

print(df.info())

print("-"*150)

print(df.describe())

print("-"*150)

features = df.drop(['cnt', 'casual', 'registered'], axis=1).copy()
labels = df['cnt'].copy()
train_X, test_X, train_y, test_y = train_test_split(features, labels, test_size=0.2, stratify=df['season'], shuffle=True, random_state=42)
print(f"""
    Train Size: {len(train_X)}
    Test Size: {len(test_X)}""")

preprocessors = preprocess() # preprocess the data (cleaning, encoding, scaling, etc.)

"""

Model Training (trying ensemble models RandomForestRegressor and GradientBoostingRegressor
with hyperparameter tuning using RandomizedSearchCV and 5-fold cross-validation)
and testing the models on the test set
and saving the models and the metrics
Hyperparameter tuning for RandomForestRegressor
Already tried Linear Regression they are performing poorly on this dataset (data is not linearly separable)
so I am trying ensemble models RandomForestRegressor and GradientBoostingRegressor 

"""


from sklearn.ensemble import RandomForestRegressor

param_grid = {
    'randomforestregressor__n_estimators': randint(100, 500),
    'randomforestregressor__max_features': [None, 'sqrt', 'log2', 0.5],
    'randomforestregressor__max_depth': [None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'randomforestregressor__min_samples_split': randint(2, 20),
    'randomforestregressor__min_samples_leaf': randint(1, 20),
    'randomforestregressor__bootstrap': [True, False]
}

model_rf = ModelTrainer(RandomForestRegressor(random_state=42), preprocessor=preprocessors, param_grid=param_grid, cv=5, n_iter=10)
model_rf.train(train_X, train_y)
model_rf.hypertune(train_X, train_y)
model_rf.test(test_X, test_y)
model_rf.save_model()
model_rf.save_metrics()

# Hyperparameter tuning for GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor

param_grid = {
    'gradientboostingregressor__n_estimators': randint(100, 500),
    'gradientboostingregressor__max_features': [None, 'sqrt', 'log2', 0.5],
    'gradientboostingregressor__max_depth': [None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'gradientboostingregressor__min_samples_split': randint(2, 20),
    'gradientboostingregressor__min_samples_leaf': randint(1, 20),
    'gradientboostingregressor__learning_rate': uniform(0.01, 0.5)
}

model_gb = ModelTrainer(GradientBoostingRegressor(random_state=42), preprocessor=preprocessors, param_grid=param_grid, cv=5, n_iter=10)
model_gb.train(train_X, train_y)
model_gb.hypertune(train_X, train_y)
model_gb.test(test_X, test_y)
model_gb.save_model()
model_gb.save_metrics()