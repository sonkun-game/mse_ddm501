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
    income = int(request.form['income'])
    education_level = int(request.form['education_level'])
    years_experience = int(request.form['years_experience'])
    credit_score = int(request.form['credit_score'])    
    # Tạo DataFrame từ dữ liệu đầu vào
    input_data = pd.DataFrame([[age, income, education_level, years_experience, credit_score]], 
    columns=['age', 'income', 'education_level', 'years_experience', 'credit_score'])
    # Dự đoán kết quả
    prediction = model.predict(input_data)
    text_display = 'Rủi ro cao'
    if prediction[0] == 1:
        text_display = 'Rủi ro thấp'
    
    return render_template('index.html', prediction_text=f'Khả năng chi trả: {text_display}')
