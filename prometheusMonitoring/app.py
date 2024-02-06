from flask import Flask, request
from prometheus_client import Counter, generate_latest, REGISTRY
import time

app = Flask(__name__)

# Define Prometheus metrics
login_counter = Counter('user_logins_total', 'Total number of user logins')
active_sessions = Counter('active_sessions', 'Number of active user sessions')

@app.route('/login', methods=['POST'])
def login():
    # Perform login logic
    username = request.json.get('username')
    # Example: Assuming successful login
    login_counter.inc()
    active_sessions.inc()
    return 'Logged in successfully'

@app.route('/logout', methods=['POST'])
def logout():
    # Perform logout logic
    username = request.json.get('username')
    # Example: Assuming successful logout
    active_sessions.dec()
    return 'Logged out successfully'

@app.route('/metrics')
def metrics():
    # Expose Prometheus metrics
    return generate_latest(REGISTRY)

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
