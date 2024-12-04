from flask import request, jsonify, render_template, Blueprint
import joblib
import pandas as pd  

import joblib

model = joblib.load('models/model.pkl')

main = Blueprint('main', __name__)

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    education = int(request.form['education'])
    frequency = int(request.form['frequency'])
    platform = int(request.form['platform'])
    is_addicted = int(request.form['is_addicted']) 
    # Tạo dữ liệu đầu vào
    input_data = pd.DataFrame([[age, education, frequency, platform, is_addicted]], 
    columns=['age', 'education', 'frequency', 'platform', 'is_addicted'])
    # Dự đoán kết quả
    prediction = model.predict(input_data)
    text_display = 'Rủi ro cao'
    if prediction[0] == 1:
        text_display = 'Rủi ro thấp'
    
    return render_template('index.html', prediction_text=f'Khả năng nghiện sử dụng nền tảng xã hội: {text_display}')

