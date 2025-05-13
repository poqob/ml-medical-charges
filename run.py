# filepath: /mnt/newdisk/dosyalar/Dosyalar/projeler/py/ML/medical-charges/run.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

# Çıktı klasörünü kontrol et ve oluştur
os.makedirs('output', exist_ok=True)

# Veri setini yükleme
print("Veri seti yükleniyor...")
df = pd.read_csv('data/medical-charges.csv')

# Veri seti hakkında genel bilgi
print("\nVeri seti boyutu:", df.shape)
print("\nVeri seti bilgileri:")
print(df.info())
print("\nİstatistiksel özet:")
print(df.describe())

# İlk 5 satırı göster
print("\nİlk 5 satır:")
print(df.head())

# Eksik değerleri kontrol et
print("\nEksik değerler:")
print(df.isnull().sum())

# Korelasyon analizi
print("\nKorelasyon analizi:")
correlation = df.corr()
print(correlation['charges'].sort_values(ascending=False))

# Görselleştirmeler
plt.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Korelasyon Matrisi')
plt.savefig('output/korelasyon_matrisi.png')

# Hedef değişkenin dağılımı
plt.figure(figsize=(10, 6))
sns.histplot(df['charges'], kde=True)
plt.title('Sigorta Ücretlerinin Dağılımı')
plt.xlabel('Ücretler')
plt.savefig('output/ucretlerin_dagilimi.png')

# Yaş ile ücretler arasındaki ilişki
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='charges', hue='smoker', data=df)
plt.title('Yaş ve Sigara Kullanımına Göre Sigorta Ücretleri')
plt.savefig('output/yas_sigara_ucretler.png')

# BMI ile ücretler arasındaki ilişki
plt.figure(figsize=(10, 6))
sns.scatterplot(x='bmi', y='charges', hue='smoker', data=df)
plt.title('BMI ve Sigara Kullanımına Göre Sigorta Ücretleri')
plt.savefig('output/bmi_sigara_ucretler.png')

print("Veri analizi tamamlandı ve grafikler 'output' klasörüne kaydedildi.")