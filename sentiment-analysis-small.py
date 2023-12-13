import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from transformers import DistilBertTokenizer, TFDistilBertModel
from tensorflow.keras.optimizers import Adam
from textblob import Word
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk 
from nltk.corpus import stopwords
import snowballstemmer 
import gc  # Bellek temizleme için

# Veri seti "Metin" ve "Durum" sütunlarına sahip olduğu için
df= pd.read_csv("e-ticaret_urun_yorumlari.csv", delimiter=';')
print(df.head())

# Metin temizleme
df['Yeni Metin'] = df['Metin']
# Büyük-küçük dönüşümü
df['Yeni Metin'] = df['Yeni Metin'].apply(lambda x: " ".join(x.lower() for x in x.split()))
# Noktalama işaretleri
df['Yeni Metin'] = df['Yeni Metin'].str.replace('[^\w\s\.]', '')
# Sayılar
df['Yeni Metin'] = df['Yeni Metin'].str.replace('\d', '')
# Stopwords
sw = stopwords.words('turkish')
df['Yeni Metin'] = df['Yeni Metin'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

# Temizlenen veriyi yazdır
print(df[['Metin', 'Yeni Metin']][:10])




# Stemming with snowballstemmer
stemmer = snowballstemmer.stemmer('turkish')
print(stemmer.stemWords("teşekkürler".split()))

df["Kök Metin"]=df["Yeni Metin"].apply(lambda x: " ".join([stemmer.stemWord(word) for word in x.split()]))
df[["Yeni Metin",'Kök Metin']][:10]






#TF-IDF
tf= (df["Kök Metin"]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf


#Vektörleştirme
vectorizer = TfidfVectorizer()
Xtf = vectorizer.fit_transform(df["Kök Metin"][:10])

tokens = vectorizer.get_feature_names()

tf_idf = pd.DataFrame(data = Xtf.toarray(), columns = tokens)

print(tf_idf)

#Duygu Analizi

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn import naive_bayes

train_x, test_x, train_y, test_y = model_selection.train_test_split(tf_idf,  df["Durum"],test_size=0.30,random_state = 42 )



nb_multi = naive_bayes.MultinomialNB()
nb_model_multi = nb_multi.fit(train_x,train_y)
y_pred_nb_multi=nb_model_multi.predict(test_x)
print("Multinominal Accuracy:", accuracy_score(test_y, y_pred_nb_multi, normalize=True))
print(classification_report(test_y, y_pred_nb_multi))

y_pred_nb_multi_train=nb_model_multi.predict(train_x)
print("Multinominal train Accuracy:", accuracy_score(train_y, y_pred_nb_multi_train, normalize=True))
print(classification_report(train_y, y_pred_nb_multi_train))

print("------------------------------------------------------------------------------------------")

print("test cross val skor:" , cross_val_score(nb_model_multi, test_x, test_y, cv = 10).mean())

print("train cross val skor:" ,cross_val_score(nb_model_multi, train_x, train_y, cv = 10).mean())






#Yeni Yorum Duygu Tahmini

yeni_yorum = pd.Series("168 boyum 83 kg yum gerçekten üzerime şahane oldu ve çok beğendim sadece yaka kısmı bana göre biraz açık onada iğne yada gizli bir dikişle halledicem  kalitesi kumaşı herseyiyle çok güzel teşekkürler") #olumlu

yeni_yorum2 = pd.Series("Yanlış elbise göndermişler. Bu elbiseyle alakası yok") #olumsuz

yeni_yorum3 = pd.Series("az kalın olabilirdi") #nötr
yeni_yorum = vectorizer.transform(yeni_yorum)
yeni_yorum


nb_model_multi.predict(yeni_yorum)

yeni_yorum2 = vectorizer.transform(yeni_yorum2)
yeni_yorum2

nb_model_multi.predict(yeni_yorum2)

yeni_yorum3 = vectorizer.transform(yeni_yorum3)
yeni_yorum3

nb_model_multi.predict(yeni_yorum3)





# Train ve test setleri
#X_train, X_test, y_train, y_test = train_test_split(df['Metin'], df['Durum'], test_size=0.2, random_state=42)

# Tokenizer'ı oluşturma ve eğitim verilerini tokenleştirme
'''max_words = 10000  # en sık kullanılan 10,000 kelimeyi kullandık
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# Eğitim verilerini ve test verilerini tokenleştirme
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Veri setini aynı uzunluğa getirilmesi (padding)
max_len = max(len(x) for x in X_train_seq)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

# Bellek temizleme
gc.collect()'''

'''
# BERT
tokenizer_bert = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
model_bert = TFDistilBertModel.from_pretrained('distilbert-base-multilingual-cased')

# BERT tokenleme ve özellik çıkarma
X_train_bert = tokenizer_bert(X_train.tolist(), padding=True, truncation=True, return_tensors="tf")
X_test_bert = tokenizer_bert(X_test.tolist(), padding=True, truncation=True, return_tensors="tf")

# Bellek temizleme
gc.collect()


# BERT çıkarma işlemini uygula
X_train_bert = model_bert(X_train_bert.input_ids).last_hidden_state[:, 0, :]
X_test_bert = model_bert(X_test_bert.input_ids).last_hidden_state[:, 0, :]

# Bellek temizleme
gc.collect()

# Modeli oluşturma ve eğitme
model_bert = Sequential()
model_bert.add(Dense(64, activation='relu', input_shape=(768,)))
model_bert.add(Dense(1, activation='sigmoid'))
model_bert.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

model_bert.fit(X_train_bert, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Modelin değerlendirilmesi
y_pred_bert = (model_bert.predict(X_test_bert) > 0.5).astype("int32")

# Metrikler
print("\nBERT Confusion Matrix:\n", confusion_matrix(y_test, y_pred_bert))
print("\nBERT Classification Report:\n", classification_report(y_test, y_pred_bert))
print("\nBERT Accuracy:", accuracy_score(y_test, y_pred_bert))

'''


# LSTM Modeli
'''model_lstm = Sequential()
model_lstm.add(Embedding(max_words, 64, input_length=max_len))
model_lstm.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model_lstm.add(Dense(1, activation='sigmoid'))

model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_lstm.summary()

# LSTM Modelini Eğitme
model_lstm.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_split=0.1)

#eğitim işlemi sırasında doğruluk değerlerini kontrol etme
history = model_lstm.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_split=0.1)
print("Eğitim Doğruluk Değerleri:", history.history['accuracy'])
print("Doğrulama Doğruluk Değerleri:", history.history['val_accuracy'])


# Test seti üzerinde modelin performansını değerlendirme
loss, accuracy = model_lstm.evaluate(X_test_pad, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# LSTM Modeli Değerlendirme
y_pred_lstm = (model_lstm.predict(X_test_pad) > 0.5).astype("int32")


# Metrikler
print("\nLSTM Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lstm))
print("\nLSTM Classification Report:\n", classification_report(y_test, y_pred_lstm))
print("\nLSTM Accuracy:", accuracy_score(y_test, y_pred_lstm))

# Yeni örnekleri tahmin etme
new_texts = ['Excellent movie!', 'Terrible experience.']
new_texts_seq = tokenizer.texts_to_sequences(new_texts)
new_texts_pad = pad_sequences(new_texts_seq, maxlen=max_len)
predictions = model_lstm.predict(new_texts_pad)

for i, text in enumerate(new_texts):
    sentiment = 'Positive' if predictions[i] > 0.5 else 'Negative'
    print(f'Text: {text}, Predicted Sentiment: {sentiment}')'''

