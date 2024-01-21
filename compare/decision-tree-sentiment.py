import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import nltk
import snowballstemmer
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# Veriyi oku
df = pd.read_csv("e-ticaret_urun_yorumlari.csv", delimiter=';')

# Veriyi hazırla
df['Yeni Metin'] = df['Metin'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df['Yeni Metin'] = df['Yeni Metin'].str.replace('[^\w\s]', '')
df['Yeni Metin'] = df['Yeni Metin'].str.replace('\d', '')
sw = stopwords.words('turkish')
df['Yeni Metin'] = df['Yeni Metin'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

stemmer = snowballstemmer.stemmer('turkish')
df["Kök Metin"] = df["Yeni Metin"].apply(lambda x: " ".join([stemmer.stemWord(word) for word in x.split()]))

# TF-IDF vektörleme
vectorizer = TfidfVectorizer()
Xtf = vectorizer.fit_transform(df["Kök Metin"])
y = df["Durum"]

# Veriyi train ve test olarak böl
train_x, test_x, train_y, test_y = train_test_split(Xtf, y, test_size=0.2, random_state=42)

# Modeli oluştur
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(train_x, train_y)

# Tahmin yap
y_pred_dt = dt_model.predict(test_x)

# Performansı değerlendir
print("Decision Tree Accuracy:", accuracy_score(test_y, y_pred_dt, normalize=True))
print(classification_report(test_y, y_pred_dt))
cm_dt = confusion_matrix(test_y, y_pred_dt)
print(cm_dt)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Neutral", "Positive"], yticklabels=["Negative", "Neutral", "Positive"])
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Yeni yorumları tahmin et
yeni_yorum = pd.Series("168 boyum 83 kg yum gerçekten üzerime şahane oldu ve çok beğendim sadece yaka kısmı bana göre biraz açık onada iğne yada gizli bir dikişle halledicem  kalitesi kumaşı herseyiyle çok güzel teşekkürler")  # olumlu
yeni_yorum2 = pd.Series("Yanlış elbise göndermişler. Bu elbiseyle alakası yok")  # olumsuz
yeni_yorum3 = pd.Series("az kalın olabilirdi")  # nötr

yeni_yorum_tfidf = vectorizer.transform(yeni_yorum)
y_pred_yeni_yorum = dt_model.predict(yeni_yorum_tfidf)
print("Yeni Yorum (olumlu) Tahmini:", y_pred_yeni_yorum)

yeni_yorum2_tfidf = vectorizer.transform(yeni_yorum2)
y_pred_yeni_yorum2 = dt_model.predict(yeni_yorum2_tfidf)
print("Yeni Yorum (olumsuz) Tahmini:", y_pred_yeni_yorum2)

yeni_yorum3_tfidf = vectorizer.transform(yeni_yorum3)
y_pred_yeni_yorum3 = dt_model.predict(yeni_yorum3_tfidf)
print("Yeni Yorum (nötr) Tahmini:", y_pred_yeni_yorum3)
