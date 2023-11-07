import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Veri okuma
veri = pd.read_csv("veri_seti.csv")  # Veri setinin okunması

# Veri analizi
print("Veri Seti İlk 5 Satir:\n", veri.head())
print("Veri Seti Temel İstatistikler:\n", veri.describe())

# Null değer kontrolü
print("Null Değerler:\n", veri.isnull().sum())

# Veri dağılımı inceleme
for kolon in veri.columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(veri[kolon], kde=True)
    plt.title(f"{kolon} Dagilimi")
    plt.xlabel(kolon)
    plt.ylabel("Frekans")
    plt.show()

# Korelasyon ilişkisi inceleme
correlation_matrix = veri.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Değişkenler Arasindaki Korelasyon Matrisi")
plt.show()