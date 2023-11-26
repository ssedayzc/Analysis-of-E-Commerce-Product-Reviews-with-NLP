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
import gc  # Bellek temizleme için

# Veri seti "Metin" ve "Durum" sütunlarına sahip olduğu için
df = pd.read_csv("e-ticaret_urun_yorumlari.csv", delimiter=';')

# Veri setini inceleme
print(df.head())

# Train ve test setleri
X_train, X_test, y_train, y_test = train_test_split(df['Metin'], df['Durum'], test_size=0.2, random_state=42)

# Tokenizer'ı oluşturma ve eğitim verilerini tokenleştirme
max_words = 10000  # en sık kullanılan 10,000 kelimeyi kullandık
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
gc.collect()

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
