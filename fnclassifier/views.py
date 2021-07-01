from django.shortcuts import render
from django.shortcuts import render,redirect

#Deep learning model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def dmodel():
    #Defining Neural Network
    model = Sequential()
    #Non-trainable embeddidng layer
    model.add(Embedding(10000, output_dim=100, input_length=300))
    #LSTM 
    model.add(LSTM(units=128 , return_sequences = True , recurrent_dropout = 0.25 , dropout = 0.25))
    model.add(LSTM(units=64 , recurrent_dropout = 0.1 , dropout = 0.1))
    model.add(Dense(units = 32 , activation = 'relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


#import lib for preprocessing
from bs4 import BeautifulSoup
import re,string,unicodedata
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)
from nltk.stem.porter import PorterStemmer

import pickle
with open('./model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
model = dmodel()
model.load_weights('./model/model.h5')
# Create your views here.

def home(request):
    return render(request, 'home.html')

def classify(request):
    text = request.POST.get('newstext')
    text = preprocess(text)
    news = []
    news.append(text)
    testing_sequences = tokenizer.texts_to_sequences(news)
    testing_padded = pad_sequences(testing_sequences, maxlen=300, padding='post', truncating='post')
    pred = model.predict_classes(testing_padded)
    return render(request,'result.html',{'fresult':pred,'text':text,'testing_padded':testing_padded})



#preprocessing text   
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def remove_urls(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    return " ".join(final_text)


def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def stem(text):
    ps = PorterStemmer()
    review = text
    review = review.lower()
    review = review.split()    
    review = [ps.stem(word) for word in review]
    review = ' '.join(review)
    return review

def preprocess(text):
    #remove extra white spaces
    #text=re.sub(' +', ' ', text)
    text=strip_html(text)
    text=remove_between_square_brackets(text)
    text=remove_urls(text)
    text=remove_stopwords(text)
    text=remove_punct(text)
    text=remove_emoji(text)
    text=stem(text)
    return text

