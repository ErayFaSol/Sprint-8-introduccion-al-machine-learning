# src/models/logistic_regression.py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def train_logistic_regression(X_train, y_train):
    param_grid_lr = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }

    grid_search_lr = GridSearchCV(estimator=LogisticRegression(random_state=12345, max_iter=200), 
                                  param_grid=param_grid_lr, 
                                  cv=5, 
                                  scoring='accuracy')
    grid_search_lr.fit(X_train, y_train)
    return grid_search_lr.best_estimator_
