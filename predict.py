import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

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

def main():
    print("Sağlık Sigortası Ücret Tahmini Uygulaması")
    print("----------------------------------------")
    
    # Modelleri yükle
    try:
        lr_model = joblib.load('models/linear_regression_model.pkl')
        rf_model = joblib.load('models/random_forest_model.pkl')
        print("Modeller başarıyla yüklendi.")
    except FileNotFoundError:
        print("Hata: Model dosyaları bulunamadı. Lütfen modellerin oluşturulduğundan emin olun.")
        return
    
    # Örnek veri seti
    print("\nÖrnek veri seti ile tahmin yapılıyor...")
    
    # Birkaç test örneği oluşturalım
    test_samples = [
        {"age": 30, "sex": "male", "bmi": 25.0, "children": 1, "smoker": "no", "region": "northeast"},
        {"age": 45, "sex": "female", "bmi": 30.5, "children": 2, "smoker": "yes", "region": "southeast"},
        {"age": 70, "sex": "male", "bmi": 35.2, "children": 0, "smoker": "no", "region": "southwest"},
        {"age": 25, "sex": "female", "bmi": 22.0, "children": 0, "smoker": "no", "region": "northwest"},
        {"age": 50, "sex": "male", "bmi": 28.5, "children": 3, "smoker": "yes", "region": "northeast"}
    ]
    
    # Tahminleri yap ve yazdır
    print("\n{:<5} {:<15} {:<10} {:<8} {:<12} {:<10} {:<15} {:<20} {:<20}".format(
        "No", "Yaş", "Cinsiyet", "BMI", "Çocuk Sayısı", "Sigara", "Bölge", "LR Tahmini ($)", "RF Tahmini ($)"))
    print("-" * 115)
    
    for i, sample in enumerate(test_samples):
        lr_pred = predict_medical_charge(sample, lr_model)
        rf_pred = predict_medical_charge(sample, rf_model)
        
        print("{:<5} {:<15} {:<10} {:<8.1f} {:<12} {:<10} {:<15} {:<20.2f} {:<20.2f}".format(
            i+1,
            sample['age'],
            sample['sex'],
            sample['bmi'],
            sample['children'],
            sample['smoker'],
            sample['region'],
            lr_pred,
            rf_pred
        ))
    
    # Gerçek verilerle karşılaştırma
    print("\nGerçek veri üzerinde model başarımı değerlendiriliyor...")
    
    try:
        # Orijinal veri setini yükle
        df = pd.read_csv('data/medical-charges.csv')
        
        # Rastgele 20 örnek seç
        test_size = 20
        test_indices = np.random.choice(df.index, size=test_size, replace=False)
        df_test = df.loc[test_indices].copy()
        
        # Test seti için tahminleri hazırla
        test_encoded = df_test.copy()
        test_encoded['sex'] = df_test['sex'].map({'female': 0, 'male': 1})
        test_encoded['smoker'] = df_test['smoker'].map({'no': 0, 'yes': 1})
        test_encoded = pd.get_dummies(test_encoded, columns=['region'], drop_first=True)
        
        # Eğer sütunlar eksikse, ekle
        for col in ['region_northwest', 'region_southeast', 'region_southwest']:
            if col not in test_encoded.columns:
                test_encoded[col] = 0
        
        # Gerçek değerleri kaydet
        y_actual = test_encoded['charges']
        
        # Özellikleri ayır
        X_test = test_encoded.drop('charges', axis=1)
        
        # Tahminler
        lr_preds = lr_model.predict(X_test)
        rf_preds = rf_model.predict(X_test)
        
        # Metrikler
        lr_rmse = np.sqrt(mean_squared_error(y_actual, lr_preds))
        lr_r2 = r2_score(y_actual, lr_preds)
        
        rf_rmse = np.sqrt(mean_squared_error(y_actual, rf_preds))
        rf_r2 = r2_score(y_actual, rf_preds)
        
        print("\nLineer Regresyon Modeli Başarımı:")
        print(f"RMSE: ${lr_rmse:.2f}")
        print(f"R² Skoru: {lr_r2:.4f}")
        
        print("\nRandom Forest Modeli Başarımı:")
        print(f"RMSE: ${rf_rmse:.2f}")
        print(f"R² Skoru: {rf_r2:.4f}")
        
    except FileNotFoundError:
        print("Uyarı: Orijinal veri seti bulunamadığı için gerçek değerlerle karşılaştırma yapılamadı.")

if __name__ == "__main__":
    main()