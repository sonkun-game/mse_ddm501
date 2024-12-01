import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import mlflow.sklearn
import pandas as pd

data = pd.read_csv("data.csv")
X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Định nghĩa siêu tham số
param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'saga']
}

# Tạo GridSearchCV
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

#Log kết quả tốt nhất vào MLflow
best_model = grid_search.best_estimator_
best_accuracy = grid_search.best_score_

mlflow.set_tracking_uri('http://127.0.0.1:8080')


mlflow.set_experiment('Grid_Search_CV')

with mlflow.start_run() as run:
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("best_param",grid_search.best_params_)
    mlflow.log_metric("accuracy", best_accuracy)
    mlflow.sklearn.log_model(best_model,"best_model")
    run_id = run.info.run_id
    print(run_id)
