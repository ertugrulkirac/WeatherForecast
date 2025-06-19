# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 21:44:29 2025

@author: ekirac
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Veri setini yükle
df = pd.read_csv("weather_forecast_data.csv")

# Özellikler (X) ve hedef değişken (y) ayrımı
X = df.drop("Rain", axis=1)
y = df["Rain"]

# Kategorik hedef değişkeni sayısal değere dönüştür (rain = 1, no rain = 0)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Özellikleri standartlaştır
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim ve test verisini ayır (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Kullanılacak modelleri tanımla
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

# Model sonuçlarını tutacak sözlük
results = {}

# Her modeli eğit, test et ve sonuçları kaydet
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    
    results[name] = {
        "Accuracy": round(acc * 100, 2),
        "Precision (rain)": round(report["rain"]["precision"], 2),
        "Recall (rain)": round(report["rain"]["recall"], 2),
        "F1-Score (rain)": round(report["rain"]["f1-score"], 2)
    }

# Sonuçları tablo olarak yazdır
results_df = pd.DataFrame(results).T
print("Model Karşılaştırma Sonuçları:\n")
print(results_df)
