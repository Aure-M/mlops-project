import unittest
from utils import preprocess_text, getDecisionModelsJsonData, getVectorizersJsonData
import joblib
import nltk

nltk.download('stopwords')

decision_tree_modelData = getDecisionModelsJsonData()
vec_modelData = getVectorizersJsonData()

model = joblib.load('./backEnd/mlModel/versions/models/decisionTree/{}'.format(decision_tree_modelData["name"]))
tfidf_vectorizer = joblib.load('./backEnd/mlModel/versions/models/vectorizers/{}'.format(vec_modelData["name"]))



class TestPreprocessText(unittest.TestCase):
    def test_lowercase_conversion(self):
        text = "Hello World"
        processed_text = preprocess_text(text)
        self.assertEqual(processed_text, "hello world")

    def test_special_characters_removal(self):
        text = "Hello @World!"
        processed_text = preprocess_text(text)
        self.assertEqual(processed_text, "hello world")


class TestSentimentPrediction(unittest.TestCase):
    def test_sentiment_prediction_positive(self):
        response = self.app.post('/predict', json={'text': 'I love this product'})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('sentiment', data)
        self.assertEqual(data['sentiment'], 'Positive')
        
    def test_sentiment_prediction_negative(self):
          response = self.app.post('/predict', json={'text': 'I hate it this is so bad'})
          self.assertEqual(response.status_code, 200)
          data = response.get_json()
          self.assertIn('sentiment', data)
          self.assertEqual(data['sentiment'], 'Negative')
          

    def test_sentiment_prediction_neutral(self):
          response = self.app.post('/predict', json={'text': 'I hate that this easy mayhem modifier event on mayhem won’t last forever.'})
          self.assertEqual(response.status_code, 200)
          data = response.get_json()
          self.assertIn('sentiment', data)
          self.assertEqual(data['sentiment'], 'Neutral')

    def test_sentiment_prediction_irrelevant(self):
          input_text = "Another successful stream last night"
          expected_sentiment = "Irrelevant"
          response = self.app.post('/predict', json={'text': 'Another successful stream last night'})
          self.assertEqual(response.status_code, 200)
          data = response.get_json()
          self.assertIn('sentiment', data)
          self.assertEqual(data['sentiment'], 'Irrelevant')



if __name__ == '__main__':
    unittest.main()
