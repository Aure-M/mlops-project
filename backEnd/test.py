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


class TestSentimentPrediction(unittest.TestCase):
    def test_sentiment_prediction_positive(self):
        input_text = "I love this product"
        expected_sentiment = "Positive"

        processed_text = preprocess_text(input_text)

        text_vector = tfidf_vectorizer.transform([processed_text])
        predicted_sentiment = model.predict(text_vector)[0]

        self.assertEqual(predicted_sentiment, expected_sentiment)

  def test_sentiment_prediction_negative(self):
          input_text = "I hate it this is so bad"
          expected_sentiment = "Negative"
  
          processed_text = preprocess_text(input_text)
  
          text_vector = tfidf_vectorizer.transform([processed_text])
          predicted_sentiment = model.predict(text_vector)[0]
  
          self.assertEqual(predicted_sentiment, expected_sentiment)

  def test_sentiment_prediction_neutral(self):
          input_text = "I hate that this easy mayhem modifier event on mayhem won’t last forever."
          expected_sentiment = "Neutral"
  #Another successful stream last night
          processed_text = preprocess_text(input_text)
  
          text_vector = tfidf_vectorizer.transform([processed_text])
          predicted_sentiment = model.predict(text_vector)[0]
  
          self.assertEqual(predicted_sentiment, expected_sentiment)

    def test_sentiment_prediction_irrelevant(self):
          input_text = "Another successful stream last night"
          expected_sentiment = "Irrelevant"

        processed_text = preprocess_text(input_text)
  
          text_vector = tfidf_vectorizer.transform([processed_text])
          predicted_sentiment = model.predict(text_vector)[0]
  
          self.assertEqual(predicted_sentiment, expected_sentiment)


if __name__ == '__main__':
    unittest.main()
