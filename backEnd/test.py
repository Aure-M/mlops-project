import unittest
from flask_testing import TestCase
from utils import preprocess_text, getDecisionModelsJsonData, getVectorizersJsonData
import joblib
import nltk
from app import app

nltk.download('stopwords')

decision_tree_modelData = getDecisionModelsJsonData()
vec_modelData = getVectorizersJsonData()

print(decision_tree_modelData["name"])
print(vec_modelData["name"])
model = joblib.load('./mlModel/versions/models/decisionTree/{}'.format(decision_tree_modelData["name"]))
tfidf_vectorizer = joblib.load('./mlModel/versions/models/vectorizers/{}'.format(vec_modelData["name"]))


sentimentMap = {
        "Irrelevant": 0, "Negative": 1, "Neutral" : 2, "Positive": 3
    }


class TestPreprocessText(unittest.TestCase):
    def test_lowercase_conversion(self):
        text = "Hello World"
        processed_text = preprocess_text(text)
        self.assertEqual(processed_text, "hello world")

    def test_special_characters_removal(self):
        text = "Hello @World!"
        processed_text = preprocess_text(text)
        self.assertEqual(processed_text, "hello world")


class TestSentimentPrediction(TestCase):
    def create_app(self):
        return app
    def test_sentiment_prediction_positive(self):
        response = self.client.post('/predict', json={'text': 'I love this product'})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('sentiment', data)
        self.assertEqual(data['sentiment'], sentimentMap['Positive'])
        
    def test_sentiment_prediction_negative(self):
          response = self.client.post('/predict', json={'text': 'I hate it this is so bad'})
          self.assertEqual(response.status_code, 200)
          data = response.get_json()
          self.assertIn('sentiment', data)
          self.assertEqual(data['sentiment'], sentimentMap['Negative'])
          

    def test_sentiment_prediction_neutral(self):
          response = self.client.post('/predict', json={'text': 'I hate that this easy mayhem modifier event on mayhem won’t last forever.'})
          self.assertEqual(response.status_code, 200)
          data = response.get_json()
          self.assertIn('sentiment', data)
          self.assertEqual(data['sentiment'], sentimentMap['Neutral'])

    def test_sentiment_prediction_irrelevant(self):
          input_text = "Another successful stream last night"
          expected_sentiment = "Irrelevant"
          response = self.client.post('/predict', json={'text': 'How the hell are we into Halloween month already?! . '})
          self.assertEqual(response.status_code, 200)
          data = response.get_json()
          self.assertIn('sentiment', data)
          self.assertEqual(data['sentiment'], sentimentMap['Irrelevant'])



if __name__ == '__main__':
    unittest.main()
