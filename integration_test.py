import unittest
import requests

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.base_url = 'http://localhost:5000'  

    def test_login(self):
        login_response = requests.post(f'{self.base_url}/login', json={'password': 'root'})
        self.assertEqual(login_response.status_code, 200)

        login_response_incorrect = requests.post(f'{self.base_url}/login', json={'password': 'password'})
        self.assertEqual(login_response_incorrect.status_code, 400)

    def test_sentiment_analysis(self):
        session = requests.Session()
        session.post(f'{self.base_url}/login', json={'password': 'root'})

        texts = [
            'I love this product',
            'I hate it this is so bad',
            'I hate that this easy mayhem modifier event on mayhem won’t last forever.',
            'Another successful stream last night'
        ]

        for text in texts:
            response = session.post(f'{self.base_url}/predict', json={'text': text})
          #je ne suis pas sur du base_url ....
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn('sentiment', data)


if __name__ == '__main__':
    unittest.main()
