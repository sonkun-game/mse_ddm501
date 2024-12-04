# pip install mlflow

from sklearn.datasets import make_classification
import numpy as np
import pandas as pd

print("--------------START-------------\n")
print('Auto generate datasets from make_classification... \n')

# Tạo dữ liệu
X, y = make_classification(
    n_samples=1000, # 1000 mẫu dữ liệu
    n_features=5, # mỗi mẫu sẽ có 4 đặc trưng
    n_informative=3, # trong số 10 đặc trưng thì sẽ có 3 đặc trưng có thông tin hữu ích 
    n_redundant=0, # Có 0 đặc trưng dư thừa,
    n_repeated=0, # Không có đặc trưng nào lặp lại
    n_classes=2, # Phân loại nhị phân
    random_state=42 # Đảm bảo rằng việc tạo ra dữ liệu là ngẫu nhiên nhưng có thể tái lập được
)

# Đặt tên cho các feature
feature_nm = ["age", "education", "frequency", "platform","is_addicted"]
data = pd.DataFrame(X, columns=feature_nm)
data['target'] = y

data['age'] = np.clip(data['age'], 17, 70).astype(int)
data['education'] = np.clip(data['education'], 1, 3).astype(int)
data['frequency'] = np.clip(data['frequency'], 1, 24).astype(int)
data['platform'] = np.clip(data['platform'], 0, 20).astype(int)
data['is_addicted'] = np.clip(data['is_addicted'], 0, 1).astype(int)


# Lưu thông tin vào file excel
data.to_csv('analysis.csv',index=False)
print(data.head())
