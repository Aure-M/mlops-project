from flask import Flask, request, jsonify
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import joblib
import psutil
from prometheus_client import CollectorRegistry, Gauge, Counter, pushadd_to_gateway, generate_latest


app = Flask(__name__)

# Setting Prometheus metrics
registry = CollectorRegistry()
memory_usage_metric = Gauge('memory_usage_bytes', 'Memory usage in bytes', registry=registry)
cpu_usage_metric = Gauge('cpu_usage_percent', 'CPU usage percentage', registry=registry)
request_counter = Counter('requests_received', 'Number of requests received by type', ['status'], registry=registry)
predictionsCounter = Counter('predictions_values', 'Number of predictions per type', ['label'], registry=registry)
logged_in_users_metric = Counter('logged_in_users', 'Number of logged-in users', registry=registry)



# Load the pre-trained model

sentimentMap = {
    0 : "Irrelevant", 1 : "Negative", 2 : "Neutral", 3 : "Positive"
}

print("Loading Decision Tree model")
model = joblib.load('./mlModel/versions/models/model.pkl')
print("Loading Vectorizer model")
tfidf_vectorizer = joblib.load('./mlModel/versions/models/vectorizer.joblib')

english_stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


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



@app.route('/metrics')
def returnRegistry():
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_usage_metric.set(cpu_percent)

    return generate_latest(registry), 200, {'Content-Type': 'text/plain; version=0.0.4'}


@app.route('/login', methods=['POST'])
def login():
    try:
        password = request.get_json()['password']
        if password == "root":

            logged_in_users_metric.inc()
            request_counter.labels(200).inc()
            return jsonify({"message":"Logged"}), 200
        else:
            request_counter.labels(400).inc()
            return jsonify({"message":"Wrong password"}), 400
    except:
        request_counter.labels(500).inc()
        return jsonify({"message":"Login failed"}), 500

@app.route('/predict', methods=['POST'])
def predict_sentiment():

    try:
        text = request.get_json()['text']
        
        # Preprocess the input text
        processed_text = preprocess_text(text)
        # Transform the preprocessed text using the TfidfVectorizer
        text_vector = tfidf_vectorizer.transform([processed_text])
        # Perform sentiment analysis using the pre-trained model
        predicted_sentiment = model.predict(text_vector)[0]

        predictionsCounter.labels(sentimentMap[int(predicted_sentiment)]).inc() # Increment the prediction count
        request_counter.labels(200).inc()
        return jsonify({"sentiment":int(predicted_sentiment)})
    except:
        request_counter.labels(500).inc()
        




if __name__ == '__main__':
    app.run(debug=True)
