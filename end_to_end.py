from streamlit_e2e_testing import StreamlitE2ETestCase, e2e_streamlit_runner
import time

class TestStreamlitApp(StreamlitE2ETestCase):
    def test_login_and_sentiment_analysis(self):
        self.send_input('Your email', 'test@gmail.com')
        self.send_input('Password', 'root')  
        self.click('Log in')

        time.sleep(2) 

        self.send_input('Text to process', 'I love this game')
        self.click('Validate')

        time.sleep(3)  

        self.assertInDOM('Result:', timeout=7)  

if __name__ == "__main__":
    e2e_streamlit_runner.run_tests()
