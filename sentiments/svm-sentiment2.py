import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Veriyi oku
df = pd.read_csv("e-ticaret_urun_yorumlari.csv", delimiter=';')

# Veriyi hazırla
df['Yeni Metin'] = df['Metin'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df['Yeni Metin'] = df['Yeni Metin'].str.replace('[^\w\s]', '')
df['Yeni Metin'] = df['Yeni Metin'].str.replace('\d', '')

# TF-IDF vektörleme
vectorizer = TfidfVectorizer()
Xtf = vectorizer.fit_transform(df["Yeni Metin"])
y = df["Durum"]

# Veriyi train ve test olarak böl
train_x, test_x, train_y, test_y = train_test_split(Xtf, y, test_size=0.30, random_state=42)

# Modeli oluştur
nb_model = MultinomialNB()
nb_model.fit(train_x, train_y)

# Tahmin yap
y_pred_nb = nb_model.predict(test_x)

# Performansı değerlendir
print("Naive Bayes Accuracy:", accuracy_score(test_y, y_pred_nb, normalize=True))
print(classification_report(test_y, y_pred_nb))
cm_nb = confusion_matrix(test_y, y_pred_nb)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_nb, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Neutral", "Positive"], yticklabels=["Negative" , "Neutral","Positive" ])
plt.title("Confusion Matrix - Naive Bayes")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# PCA uygula
pca = PCA(n_components=100)  # Belirli bir bileşen sayısı belirtin
Xtf_pca = pca.fit_transform(Xtf.toarray())

# Veriyi normalize et
scaler = StandardScaler()
Xtf_pca_normalized = scaler.fit_transform(Xtf_pca)

# Veriyi train ve test olarak böl
train_x, test_x, train_y, test_y = train_test_split(Xtf_pca_normalized, y, test_size=0.20, random_state=42)

# Modeli oluştur
nb_model_pca = MultinomialNB()
nb_model_pca.fit(train_x, train_y)

# Tahmin yap
y_pred_nb_pca = nb_model_pca.predict(test_x)

# Performansı değerlendir
print("Naive Bayes with PCA Accuracy:", accuracy_score(test_y, y_pred_nb_pca, normalize=True))
print(classification_report(test_y, y_pred_nb_pca))
cm_nb_pca = confusion_matrix(test_y, y_pred_nb_pca)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_nb_pca, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Neutral", "Positive"], yticklabels=["Positive" , "Neutral", "Negative"])
plt.title("Confusion Matrix - Naive Bayes with PCA")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Yeni yorumları tahmin et
yeni_yorum = pd.Series("168 boyum 83 kg yum gerçekten üzerime şahane oldu ve çok beğendim sadece yaka kısmı bana göre biraz açık onada iğne yada gizli bir dikişle halledicem  kalitesi kumaşı herseyiyle çok güzel teşekkürler")  # olumlu
yeni_yorum2 = pd.Series("Yanlış elbise göndermişler. Bu elbiseyle alakası yok")  # olumsuz
yeni_yorum3 = pd.Series("az kalın olabilirdi")  # nötr
yeni_yorum_tfidf = vectorizer.transform(yeni_yorum)
y_pred_yeni_yorum = nb_model.predict(yeni_yorum_tfidf)
print("Yeni Yorum (olumlu) Tahmini:", y_pred_yeni_yorum)

yeni_yorum2_tfidf = vectorizer.transform(yeni_yorum2)
y_pred_yeni_yorum2 = nb_model.predict(yeni_yorum2_tfidf)
print("Yeni Yorum (olumsuz) Tahmini:", y_pred_yeni_yorum2)

yeni_yorum3_tfidf = vectorizer.transform(yeni_yorum3)
y_pred_yeni_yorum3 = nb_model.predict(yeni_yorum3_tfidf)
print("Yeni Yorum (nötr) Tahmini:", y_pred_yeni_yorum3)

print("Naive Bayes with PCA Confusion Matrix:", cm_nb_pca)