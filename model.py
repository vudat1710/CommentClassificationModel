from importlib import reload
from imp import reload
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from langdetect import detect

def create_data_and_labels(pos_path, neg_path):
    pos_reviews, neg_reviews = [], []
    with open(pos_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            a = line.split(' ')
            if len(a) != 1:
                try:
                    lang = detect(line)
                    if lang == "en":
                        pos_reviews.append(line)
                except:
                    pos_reviews.append(line)
    with open(neg_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            a = line.split(' ')
            if len(a) != 1:
                try:
                    lang = detect(line)
                    if lang == "en":
                        neg_reviews.append(line)
                except:
                    neg_reviews.append(line)
    reviews  = pd.concat([
        pd.DataFrame({'review': pos_reviews, 'label': 1}),
        pd.DataFrame({'review': neg_reviews, 'label': 0})
    ], ignore_index=True).sample(frac=1, random_state=1)

    return reviews

df = create_data_and_labels('/home/vudat1710/Downloads/positive_comment.txt', '/home/vudat1710/Downloads/negative_comment.txt')

stop_words = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = re.sub('.*♥+.*', 'fuck', text)
    text = re.sub('.*♡+.*', ' ', text)
    text = re.sub('.*🍌+.*', ' ', text) 
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text

df['Processed_Reviews'] = df.review.apply(lambda x: clean_text(x))
df.Processed_Reviews.apply(lambda x: len(x.split(" "))).mean()
max_features = 10000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(df['Processed_Reviews'])
# list_tokenized_train = tokenizer.texts_to_sequences(df['Processed_Reviews'])

maxlen = 150
# X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
# y = df['label']

embed_size = 128
model = Sequential()
model.add(Embedding(max_features, embed_size))
model.add(Bidirectional(LSTM(32, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# checkpoint = ModelCheckpoint(
#             'model_improvement-{epoch:02d}-{val_acc:.2f}.h5',
#             monitor='val_acc',
#             mode='max',
#             save_best_only=True
#         )
# early = EarlyStopping(monitor='val_acc', mode='max', patience=5)
# batch_size = 100
# epochs = 50
# model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.3, callbacks=[early, checkpoint])
model.load_weights('/home/vudat1710/Downloads/Courses/DDS/model_abc/model_improvement-16-0.90.h5')
df_test = create_data_and_labels('/home/vudat1710/Downloads/Courses/DDS/NewModel/test_pos.txt', '/home/vudat1710/Downloads/Courses/DDS/NewModel/test_neg.txt')
df_test['review'] = df_test.review.apply(lambda x: clean_text(x))
y_test = df_test['label']
list_sentences_test = df_test["review"]
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
prediction = model.predict(X_te)
y_pred = (prediction > 0.5)
from sklearn.metrics import f1_score, confusion_matrix
print('F1-score: {0}'.format(f1_score(y_pred, y_test)))
print('Confusion matrix:')
confusion_matrix(y_pred, y_test)
