from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
import numpy as np
import joblib
import os
import json

class ModelTrainer:
    def __init__(self, model, preprocessor=None, param_grid=None, cv=5, **kwargs):
        """_summary_

        Args:
            model (model like RandomForestRegression) : Model to be trained
            preprocessor (instance of preprocessor.py) : preprocessing the raw data.
            param_grid (dictionary of Parm grid/Hyperparameters for RandomizedSearchCV) : hyperparameters to be tuned
            cv (int, optional) : cross validation splits. Defaults to 5.
            can take additional keyword arguments for RandomizedSearchCV
        """
        self.model = model
        self.preprocessor = preprocessor
        self.param_grid = param_grid
        self.cv = cv
        self.clf = make_pipeline(preprocessor, model) if preprocessor else model
        self.best_model = None
        self.metrics = {}
    
    def train(self, train_X, train_y):
        """_summary_

        Args:
            train_X (pd.DataFrame): features of the training set
            train_y (pd.DataFrame): labels of the training set

        Returns:
            None: if error occurs while fitting the model
            prints: the cross validation scores
            Saves the model name and cross validation score in the metrics dictionary
        """
        if self.preprocessor is None:
            print("No Preprocessor Provided..., Training the Model without Preprocessor")
            try:
                print(f"Training the Model: {self.model}")
                self.clf.fit(train_X, train_y)
            except Exception as e:
                print(f"Error While fitting the Model : {e}")
                return None
        else:    
            try:
                print(f"Training the Model: {self.model.__class__.__name__}")
                self.clf.fit(train_X, train_y)
            except Exception as e:
                print(f"Error While fitting the Model : {e}")
                return None
        cross_val_scores = cross_val_score(self.clf, train_X, train_y, cv=self.cv).mean()
        print(f"Cross Validation Scores ({self.cv}): {cross_val_scores}")
        self.metrics['Model Name'] = self.model.__class__.__name__
        self.metrics['cross_val_score'] = cross_val_scores.mean()

    def hypertune(self, train_X, train_y, **kwargs):
        """
        Hyperparameter Tuning using RandomizedSearchCV

        Args:
            train_X (pd.DataFrame): features of the training set
            train_y (pd.DataFrame): labels of the training set

        Returns:
            None : if no hyperparameters are provided
        
            prints: the best hyperparameters and the best score
            
            Saves the best hyperparameters and the best score in the metrics dictionary
        
        """
        if self.param_grid is None:
            print("No ParamGrid provided, No hyperparameters to tune")
            return None
        else:
            try:
                print(f"Starting Hyperparameter Tuning for {self.model.__class__.__name__}")
                self.best_model = RandomizedSearchCV(
                    self.clf,
                    self.param_grid,
                    cv=self.cv,
                    n_iter=10,
                    n_jobs=-1,
                    random_state=42,
                    **kwargs
                )
                self.best_model.fit(train_X, train_y)
            except Exception as e:
                print(f"Error While fitting the Model : {e}")
                return None
            print(f"Best Parameters: {self.best_model.best_params_}")
            print(f"Best Score: {self.best_model.best_score_}")
            self.metrics['Best Params'] = self.best_model.best_params_
            self.metrics['Best Score'] = self.best_model.best_score_
            self.best_model = self.best_model.best_estimator_

    def test(self, test_X, test_y):
        """
        test the model on the test set

        Args:
            test_X (pd.DataFrame): features of the test set
            test_y (pd.DataFrame): labels of the test set
        Returns:
            None: if no model is trained yet
            prints: the RMSE and the test score
            Saves the RMSE and the test score in the metrics dictionary
        """
        if self.best_model is None:
            print("Model Trained is Not HyperTuned Yet")
            rmse = np.sqrt(mean_squared_error(test_y, self.clf.predict(test_X)))
            print(f"Test Size: {len(test_X)}")
            print(f"RMSE: {rmse}")
            test_score = self.clf.score(test_X, test_y)
            print(f"Test Score: {test_score}")
            self.metrics['Test Score'] = test_score
            self.metrics['RMSE_test'] = rmse
        else:
            rmse = np.sqrt(mean_squared_error(test_y, self.best_model.predict(test_X)))
            print(f"Test Size: {len(test_X)}")
            print(f"RMSE: {rmse}")
            test_score = self.best_model.score(test_X, test_y)
            print(f"Test Score: {test_score}")
            self.metrics['Test Score'] = test_score
            self.metrics['RMSE_test'] = rmse
            
    # To save the trained Model        
    def save_model(self, filepath="Trained_model"):
        """
        Save the model to disk.

        Args:
            filepath (str, optional): Path to save the model file. Defaults to "Trained_model".
            
        """
        if not os.path.exists(filepath):
            os.makedirs(filepath, exist_ok=True)
        filepath = os.path.join(filepath, f"{self.model.__class__.__name__}.pkl")
        joblib.dump(self.best_model, filepath)
        print(f"Model saved as {filepath}") 
        

    def save_metrics(self, filepath="Metrics"):
        """
        Save the metrics to disk.

        Args:
            filepath (str, optional): Path to save the metrics file. Defaults to "Metrics".
        """
        if not os.path.exists(filepath):
            os.makedirs(filepath, exist_ok=True)
        filepath = os.path.join(filepath, f"{self.model.__class__.__name__}.json")
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f)
        
        print(f"Metrics saved as {filepath}")
