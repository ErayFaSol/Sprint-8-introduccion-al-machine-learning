# src/models/random_forest.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_random_forest(X_train, y_train):
    param_grid_rf = {
        'n_estimators': [100, 150, 200],
        'max_depth': [5, 10, None],
        'min_samples_leaf': [1, 4, 7],
        'min_samples_split': [2, 5, 10]
    }

    grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=12345), 
                                  param_grid=param_grid_rf, 
                                  cv=5, 
                                  scoring='accuracy')
    grid_search_rf.fit(X_train, y_train)
    return grid_search_rf.best_estimator_
