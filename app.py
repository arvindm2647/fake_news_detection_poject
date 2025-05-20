from flask import Flask, render_template, request, jsonify
import joblib
import string
from nltk.corpus import stopwords
import nltk
import os

# Initialize Flask app
app = Flask(__name__)

# Download NLTK resources
nltk.download('stopwords')

# Constants
MODEL_DIR = 'app'
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')

def clean_text(text):
    """Clean and preprocess text"""
    text = str(text).lower()
    text = "".join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

def load_model():
    """Load the model and vectorizer"""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError("Model files not found. Please train the model first.")
    
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

# Load model at startup
try:
    model, vectorizer = load_model()
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model_loaded = False

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get text from form
        text = request.form.get('news_text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Clean and vectorize the input text
        cleaned_text = clean_text(text)
        text_tfidf = vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction = model.predict(text_tfidf)
        probabilities = model.predict_proba(text_tfidf)
        
        # Prepare response
        result = {
            'prediction': 'Real' if prediction[0] == 1 else 'Fake',
            'confidence': float(probabilities[0][prediction[0]]),
            'probabilities': {
                'Fake': float(probabilities[0][0]),
                'Real': float(probabilities[0][1])
            },
            'text_sample': text[:200] + '...' if len(text) > 200 else text
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Clean and vectorize the input text
        cleaned_text = clean_text(text)
        text_tfidf = vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction = model.predict(text_tfidf)
        probabilities = model.predict_proba(text_tfidf)
        
        # Prepare response
        result = {
            'prediction': int(prediction[0]),
            'probabilities': {
                'fake': float(probabilities[0][0]),
                'real': float(probabilities[0][1])
            },
            'text': text
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)