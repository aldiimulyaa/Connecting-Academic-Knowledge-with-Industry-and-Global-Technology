import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print("==================================================")
print(" STUDI KASUS: ANALISIS KEUANGAN & PREDIKSI LATE PAYMENT")
print(" PT MAJU BERSAMA")
print("==================================================\n")

def generate_dataset():
    np.random.seed(42)
    n_samples = 1000  
    
    data = {
        'payment_method_credit': np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7]),
        'invoice_value': np.random.randint(500000, 15000000, size=n_samples),
        'customer_history_order': np.random.randint(1, 25, size=n_samples),
        'operating_expense_ratio': np.random.uniform(0.1, 0.45, size=n_samples),
        'late_payment': np.random.choice([0, 1], size=n_samples, p=[0.82, 0.18]) # 1 = Terlambat > 30 hari
    }
    return pd.DataFrame(data)

df = generate_dataset()
print(f"Total Baris Data Transaksi: {len(df)}")
print("Contoh 5 Data Teratas:")
print(df.head(), "\n")

X = df[['payment_method_credit', 'invoice_value', 'customer_history_order', 'operating_expense_ratio']]
y = df['late_payment']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print((f"Jumlah Data Training: {len(X_train)} baris"))
print(f"Jumlah Data Testing: {len(X_test)} baris\n")

print("--- Melatih Model Machine Learning ---")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)

print("Pelatihan Model Selesai!\n")

print("==================================================")
print(" EVALUASI PERFORMA MODEL")
print("==================================================")

y_pred_rf = rf_model.predict(X_test)
print("\n[ Model: Random Forest ]")
print(f"Akurasi: {accuracy_score(y_test, y_pred_rf):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_rf, zero_division=0))

y_pred_lr = lr_model.predict(X_test)
print("\n[ Model: Logistic Regression ]")
print(f"Akurasi: {accuracy_score(y_test, y_pred_lr):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_lr, zero_division=0))
print("==================================================")