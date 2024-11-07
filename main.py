import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix

# Uyarıları yok say
warnings.filterwarnings("ignore", category=FutureWarning)

# Veri setini yükle
train_data = pd.read_csv('DiseaseTraining.csv')
test_data = pd.read_csv('DiseaseTesting.csv')

# Gereksiz sütunları kaldır
train_data = train_data.drop(columns=['Unnamed: 133'], errors='ignore')

# Adım 1: Özet İstatistikler
summary_statistics = train_data.describe().T
summary_statistics[['mean', '50%', 'std', 'min', 'max']]  # Anahtar istatistikleri seçiyoruz
print("Eğitim Verisinin Özet İstatistikleri")
print(summary_statistics)

# Adım 2: Eksik Değerleri Kontrol Etme
missing_values_train = train_data.isnull().sum()
missing_values_train = missing_values_train[missing_values_train > 0]
print("Eğitim Verisindeki Eksik Değerler:")
print(missing_values_train)

# Adım 3: Aykırı Değerlerin Ele Alınması
# Z-skora göre 3'ten büyük veya -3'ten küçük değerleri aykırı kabul edip bunları medyan ile değiştirme
z_scores = train_data.iloc[:, :-1].apply(zscore)  # 'prognosis' hariç tüm sütunlara zscore uygula
outliers = (z_scores > 3) | (z_scores < -3)
for col in train_data.columns[:-1]:  # 'prognosis' hariç tüm sütunlar
    train_data.loc[outliers[col], col] = train_data[col].median()

# Adım 4: Sınıf Dengesizliği Analizi
class_distribution = train_data['prognosis'].value_counts(normalize=True)
print("Prognosis Sınıf Dağılımı:")
print(class_distribution)

# Adım 5: Görselleştirmeler
# Örnek özelliklerin histogramı
sample_features = train_data.columns[:5]  # Görselleştirmek için bazı özellikleri seçiyoruz
train_data[sample_features].hist(bins=20, figsize=(15, 10))
plt.suptitle("Özellik Dağılımları (Örneklenmiş)", fontsize=16)
plt.show()

# Korelasyon ısı haritası
plt.figure(figsize=(12, 10))
sns.heatmap(train_data.iloc[:, :-1].corr(), cmap='coolwarm', center=0, annot=False, cbar=True)
plt.title("Özelliklerin Korelasyon Isı Haritası")
plt.show()

# Sınıf dağılımı bar grafiği
plt.figure(figsize=(12, 8))
sns.countplot(y="prognosis", data=train_data, order=train_data['prognosis'].value_counts().index)
plt.title("Prognosis Sınıf Dağılımı")
plt.show()

# Adım 6: Özellik Seçimi - Düşük varyans ve yüksek korelasyonlu özelliklerin kaldırılması
numeric_features = train_data.select_dtypes(include=np.number).columns
low_variance_features = train_data[numeric_features].var()[train_data[numeric_features].var() < 0.01].index
train_data = train_data.drop(columns=low_variance_features)

# Güncellenmiş numeric_features listesi
numeric_features = train_data.select_dtypes(include=np.number).columns

# Yüksek Korelasyonlu Özelliklerin Kaldırılması
correlation_matrix = train_data[numeric_features].corr().abs()
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)]
train_data = train_data.drop(columns=high_corr_features)

# Adım 7: Özellik Dönüşümü
scaler = StandardScaler()
X = scaler.fit_transform(train_data.drop('prognosis', axis=1))
y = train_data['prognosis']

# Hedef değişkenin kodlanması (kategorik ise)
le = LabelEncoder()
y = le.fit_transform(y)

# Adım 8: Sınıf Dengesizliği Yönetimi - SMOTE Kullanımı
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

# Adım 9: Model Eğitimi ve Hiperparametre Seçimi
# Veriyi eğitim ve doğrulama için ayırma
X_train, X_val, y_train, y_val = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

# XGBoost ve LightGBM modelleri tanımlama
models = {
    'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42),
    'LightGBM': LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
}

# Adım 10: Model değerlendirme
for model_name, model in models.items():
    # Modeli eğit
    model.fit(X_train, y_train)
    # Tahmin yap
    y_pred = model.predict(X_val)

    # Tahmin olasılıklarını kontrol et
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_val)
        if y_proba.shape[1] == 2:  # İkili sınıflandırma durumunda
            y_proba = y_proba[:, 1]  # Pozitif sınıfın olasılığı
    else:
        y_proba = None

    # Sınıflandırma raporu yazdır (zero_division parametresi ile)
    print(f"\n{model_name} Sınıflandırma Raporu:")
    print(classification_report(y_val, y_pred, zero_division=0))

    # ROC-AUC hesapla ve yazdır
    if y_proba is not None:
        if len(np.unique(y_val)) > 2:  # Çok sınıflı ROC-AUC
            roc_score = roc_auc_score(y_val, y_proba, multi_class='ovr')
        else:  # İkili ROC-AUC
            roc_score = roc_auc_score(y_val, y_proba)
        print(f"{model_name} ROC-AUC Skoru: {roc_score}")

    # Karışıklık matrisi
    conf_matrix = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Karışıklık Matrisi")
    plt.xlabel("Tahmin Edilen Etiket")
    plt.ylabel("Gerçek Etiket")
    plt.show()

    # ROC Eğrisi
    if y_proba is not None and len(np.unique(y_val)) <= 2:  # Sadece ikili sınıflar için ROC eğrisi
        fpr, tpr, _ = roc_curve(y_val, y_proba)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_score:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("Yanlış Pozitif Oranı")
        plt.ylabel("Doğru Pozitif Oranı")
        plt.title(f"{model_name} ROC Eğrisi")
        plt.legend(loc="lower right")
        plt.show()

# Adım 11: Boyut İndirgeme için PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_smote)
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_smote, palette="viridis", alpha=0.6)
plt.title("SMOTE Dengelenmiş Verinin 2D PCA Görselleştirmesi")
plt.xlabel("PCA Bileşeni 1")
plt.ylabel("PCA Bileşeni 2")
plt.show()