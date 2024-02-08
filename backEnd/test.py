import unittest
from app import preprocess_text, model
import joblib

tfidf_vectorizer = joblib.load('vectorizer.pkl')


class TestPreprocessText(unittest.TestCase):
    def test_lowercase_conversion(self):
        text = "Hello World"
        processed_text = preprocess_text(text)
        self.assertEqual(processed_text, "hello world")

    def test_special_characters_removal(self):
        text = "Hello @World!"
        processed_text = preprocess_text(text)
        self.assertEqual(processed_text, "hello world")

    # Add more tests for other aspects of preprocessing like stopword removal and stemming


class TestSentimentPrediction(unittest.TestCase):
    def test_sentiment_prediction_positive(self):
        # Mock input text and expected sentiment
        input_text = "I love this product"
        expected_sentiment = "positive"

        # Preprocess the input text
        processed_text = preprocess_text(input_text)

        # Perform sentiment prediction
        text_vector = tfidf_vectorizer.transform([processed_text])
        predicted_sentiment = model.predict(text_vector)[0]

        # Assert that the predicted sentiment matches the expected sentiment
        self.assertEqual(predicted_sentiment, expected_sentiment)

  def test_sentiment_prediction_negative(self):
          # Mock input text and expected sentiment
          input_text = "I hate it this is so bad"
          expected_sentiment = "negative"
  
          # Preprocess the input text
          processed_text = preprocess_text(input_text)
  
          # Perform sentiment prediction
          text_vector = tfidf_vectorizer.transform([processed_text])
          predicted_sentiment = model.predict(text_vector)[0]
  
          # Assert that the predicted sentiment matches the expected sentiment
          self.assertEqual(predicted_sentiment, expected_sentiment)

  def test_sentiment_prediction_neutral(self):
          # Mock input text and expected sentiment
          input_text = "i am playing fifa right now"
          expected_sentiment = "neutral"
  
          # Preprocess the input text
          processed_text = preprocess_text(input_text)
  
          # Perform sentiment prediction
          text_vector = tfidf_vectorizer.transform([processed_text])
          predicted_sentiment = model.predict(text_vector)[0]
  
          # Assert that the predicted sentiment matches the expected sentiment
          self.assertEqual(predicted_sentiment, expected_sentiment)


if __name__ == '__main__':
    unittest.main()
