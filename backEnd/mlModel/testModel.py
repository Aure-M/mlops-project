import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import pandas as pd
from flask import Flask


# Load the pre-trained model
model = joblib.load('./versions/models/model.pkl')

app = Flask(__name__)

english_stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Initialize the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove some characters (e.g., @)
    text = re.sub(r'(@)', ' ', text)
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove stopwords
    text = ' '.join([
                word
                for word in text.split()
                if word not in english_stop_words
            ])
    # Perform stemming
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text


train_df = pd.read_csv(
    './data/twitter_training.csv',
    sep=',',
    names=['Tweet_ID', 'Subject', 'Sentiment', 'Tweet_content'])
train_df.dropna(axis=0, inplace=True)

train = train_df["Tweet_content"]
label = train_df['Sentiment']

# Remove some character as @
replace_by_space = re.compile("(@)")
space = " "


def preprocess_data(data):

    data = [replace_by_space.sub(space, line.lower()) for line in data]
    return data


print("preprocessing")
data_train_clean = preprocess_data(train)

# Now we are gonna remove all the stopwords


def removestopwords(data):
    removedstopwords = []
    for review in data:
        removedstopwords.append(
            ' '.join([
                word
                for word in review.split()
                if word not in english_stop_words
            ]))
    return removedstopwords


print("remove stop word")
nostopwords_train = removestopwords(data_train_clean)

# Stemming : We are gonna reduce likely words in a root word
# (ex: likes, likely, liking will be reduce in like root word)


def getstemmedtext(data):
    stemmer = PorterStemmer()

    return [' '.join([
                stemmer.stem(word)
                for word in review.split()
            ])
            for review in data]


print("get stemmed text")
stemmed_train = getstemmedtext(nostopwords_train)

# Clean Tweets' vectorization
print("vectorizer")
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(stemmed_train)


joblib.dump(tfidf_vectorizer, './versions/vectorizer.joblib')