import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Modelleri yükle
try:
    lr_model = joblib.load('models/linear_regression_model.pkl')
    rf_model = joblib.load('models/random_forest_model.pkl')
    print("Modeller başarıyla yüklendi.")
except FileNotFoundError:
    print("Hata: Model dosyaları bulunamadı.")
    exit(1)

def predict_medical_charge(sample, model):
    """
    Bir kullanıcı için sağlık sigortası ücret tahmini yapar
    
    Parametreler:
    sample -- Tahmin edilecek örnek (dict formatında)
              Örnek: {"age": 25, "sex": "male", "bmi": 22.5, 
                      "children": 0, "smoker": "no", "region": "southwest"}
    model -- Kullanılacak model
    
    Dönüş:
    float -- Tahmin edilen sigorta ücreti
    """
    # Örneği dönüştür
    sample_encoded = sample.copy()
    sample_encoded['sex'] = 1 if sample['sex'] == 'male' else 0
    sample_encoded['smoker'] = 1 if sample['smoker'] == 'yes' else 0
    
    # Bölge için one-hot encoding
    region_cols = ['region_northwest', 'region_southeast', 'region_southwest']
    for col in region_cols:
        region = col.split('_')[1]
        sample_encoded[col] = 1 if sample['region'] == region else 0
    
    # 'region' sütununu kaldır
    del sample_encoded['region']
    
    # Dataframe'e dönüştür
    df_sample = pd.DataFrame([sample_encoded])
    
    # Tahmin
    prediction = model.predict(df_sample)[0]
    return prediction

@app.route('/health', methods=['GET'])
def health_check():
    """Servisin çalışıp çalışmadığını kontrol etmek için basit bir endpoint"""
    return jsonify({"status": "up", "message": "Servis çalışıyor"})

@app.route('/predict', methods=['POST'])
def predict():
    """
    JSON formatında bir veri alarak ücret tahmini yapar
    
    Örnek istek:
    {
        "age": 30,
        "sex": "male", 
        "bmi": 25.0,
        "children": 1,
        "smoker": "no",
        "region": "northeast",
        "model": "random_forest"  # veya "linear_regression"
    }
    """
    # İstek verilerini al
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Veri bulunamadı"}), 400
    
    # Gerekli alanları kontrol et
    required_fields = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"'{field}' alanı eksik"}), 400
    
    # Seçilen modeli belirle (varsayılan olarak random forest)
    model_name = data.get('model', 'random_forest')
    
    # Model kontrolü
    if model_name not in ['random_forest', 'linear_regression']:
        return jsonify({"error": "Geçersiz model adı. 'random_forest' veya 'linear_regression' kullanın."}), 400
    
    # Kullanılacak modeli seç
    model = rf_model if model_name == 'random_forest' else lr_model
    
    # Tahmin için kullanılacak veriyi hazırla
    prediction_data = {
        'age': data['age'],
        'sex': data['sex'],
        'bmi': data['bmi'],
        'children': data['children'],
        'smoker': data['smoker'],
        'region': data['region']
    }
    
    # Tahmin yap
    try:
        prediction = predict_medical_charge(prediction_data, model)
        
        # Yanıt oluştur
        response = {
            "predicted_charge": float(prediction),
            "model_used": model_name,
            "input_data": prediction_data
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": f"Tahmin sırasında bir hata oluştu: {str(e)}"}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Toplu tahmin yapmak için kullanılan endpoint
    
    Örnek istek:
    {
        "data": [
            {"age": 30, "sex": "male", "bmi": 25.0, "children": 1, "smoker": "no", "region": "northeast"},
            {"age": 45, "sex": "female", "bmi": 30.5, "children": 2, "smoker": "yes", "region": "southeast"}
        ],
        "model": "random_forest"  # veya "linear_regression"
    }
    """
    # İstek verilerini al
    request_data = request.get_json()
    
    if not request_data or 'data' not in request_data:
        return jsonify({"error": "Veri bulunamadı veya 'data' alanı eksik"}), 400
    
    # Model seçimi
    model_name = request_data.get('model', 'random_forest')
    model = rf_model if model_name == 'random_forest' else lr_model
    
    predictions = []
    
    # Her örnek için tahmin yap
    for i, sample in enumerate(request_data['data']):
        try:
            # Gerekli alanları kontrol et
            required_fields = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
            for field in required_fields:
                if field not in sample:
                    return jsonify({"error": f"Örnek {i+1}'de '{field}' alanı eksik"}), 400
            
            # Tahmin yap
            prediction = predict_medical_charge(sample, model)
            
            # Sonucu ekle
            predictions.append({
                "input": sample,
                "predicted_charge": float(prediction)
            })
            
        except Exception as e:
            return jsonify({"error": f"Örnek {i+1} için tahmin sırasında bir hata oluştu: {str(e)}"}), 500
    
    # Yanıt oluştur
    response = {
        "model_used": model_name,
        "predictions": predictions
    }
    
    return jsonify(response)

if __name__ == "__main__":
    print("Sağlık Sigortası Ücret Tahmini API'si başlatılıyor...")
    app.run(debug=True, host='0.0.0.0', port=5000)