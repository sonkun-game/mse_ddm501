import mlflow
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv("analysis.csv")
X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_tracking_uri('http://127.0.0.1:8080')

mlflow.set_experiment('DecisionTree')

with mlflow.start_run():

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    predict = model.predict(X_test)
    accuracy = accuracy_score(y_test, predict)

    mlflow.log_param("model", "decisionTree")
    mlflow.log_metric("accuracy", accuracy)