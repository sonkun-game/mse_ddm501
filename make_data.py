# pip install mlflow

from sklearn.datasets import make_classification
import numpy as np
import pandas as pd

print("--------------START-------------\n")
print('Auto generate datasets from make_classification... \n')

# Tạo dữ liệu
X, y = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=3,
    n_redundant=1,
    n_repeated=0,
    n_classes=2,
    random_state=42
)

# Đặt tên cho các feature
feature_nm = ["age", "income", "education_level", "year_exp", "credit_score"]
data = pd.DataFrame(X, columns=feature_nm)
data['target'] = y

data['age'] = np.clip(data['age'],18,70).astype(int)
data['income'] = np.clip(data['income'], 20000, 150000).astype(int)
data['education_level'] = np.clip(data['education_level'], 0, 20).astype(int)
data['year_exp'] = np.clip(data['year_exp'], 0, 40).astype(int)
data['credit_score'] = np.clip(data['credit_score'], 300, 850).astype(int)

# Lưu thông tin vào file excel
data.to_csv('data.csv',index=False)
print(data.head())
