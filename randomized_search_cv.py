import mlflow
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import mlflow.sklearn
import numpy as np
import pandas as pd

# Định nghĩa mô hình 
model = LogisticRegression()

data = pd.read_csv("analysis.csv")
X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xác định không gian siêu tham số
param_dist = {
    'C': np.logspace(-4, 4, 20),  # Kiểm tra các giá trị của C trong khoảng từ 10^-4 đến 10^4
    'solver': ['newton-cg', 'lbfgs', 'liblinear'],  # Các solvers khác nhau để tối ưu hóa
}

# Khởi tạo RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=100, cv=5, scoring='accuracy', random_state=42)

# Tìm kiếm các tham số tốt nhất
random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_
best_accuracy = random_search.best_score_

mlflow.set_tracking_uri('http://127.0.0.1:8080')

mlflow.set_experiment('RandomizedSearchCV')

with mlflow.start_run() as run:
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("best_param",random_search.best_params_)
    mlflow.log_metric("accuracy", best_accuracy)
    mlflow.sklearn.log_model(best_model,"best_model")
    run_id = run.info.run_id
    print(run_id)