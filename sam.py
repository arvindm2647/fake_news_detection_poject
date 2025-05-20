import os
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

# Download NLTK resources
nltk.download('stopwords')
tqdm.pandas()

# Constants
DATA_PATHS = {
    'fake': 'fake_news.csv',
    'real': 'True.csv'
}
MODEL_DIR = 'app'
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')

def load_and_prepare_data():
    """Load and prepare the fake and real news datasets"""
    # Load datasets
    fake_df = pd.read_csv(DATA_PATHS['fake'])
    real_df = pd.read_csv(DATA_PATHS['real'])
    
    # Label data (0 for fake, 1 for real)
    fake_df['label'] = 0
    real_df['label'] = 1
    
    # Select relevant columns
    fake_df = fake_df[['text', 'label']]
    real_df = real_df[['text', 'label']]
    
    # Combine datasets
    df = pd.concat([fake_df, real_df], ignore_index=True)
    
    # Remove any missing values
    df.dropna(subset=['text'], inplace=True)
    
    return df

def clean_text(text):
    """Clean and preprocess text"""
    text = str(text).lower()
    text = "".join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

def train_model():
    """Train and save the fake news detection model"""
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Clean text data
    print("Cleaning text data...")
    df['text_clean'] = df['text'].progress_apply(clean_text)
    
    # Split data
    X = df['text_clean']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Vectorize text
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train model
    print("Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    
    # Save model and vectorizer
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Vectorizer saved to {VECTORIZER_PATH}")

def predict_news(text):
    """Predict whether a news article is fake or real"""
    # Load model and vectorizer
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError("Model files not found. Please train the model first.")
    
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    # Clean and vectorize the input text
    cleaned_text = clean_text(text)
    text_tfidf = vectorizer.transform([cleaned_text])
    
    # Make prediction
    prediction = model.predict(text_tfidf)
    probabilities = model.predict_proba(text_tfidf)
    
    return {
        'prediction': 'Real' if prediction[0] == 1 else 'Fake',
        'confidence': float(probabilities[0][prediction[0]]),
        'probabilities': {
            'Fake': float(probabilities[0][0]),
            'Real': float(probabilities[0][1])
        }
    }

def interactive_predictor():
    """Interactive mode for predicting news"""
    print("Fake News Detector - Interactive Mode")
    print("Enter news text (type 'quit' to exit):")
    
    while True:
        text = input("\nNews text: ")
        if text.lower() == 'quit':
            break
        
        try:
            result = predict_news(text)
            print("\nPrediction Results:")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Fake Probability: {result['probabilities']['Fake']:.2%}")
            print(f"Real Probability: {result['probabilities']['Real']:.2%}")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Train the model (uncomment if you need to retrain)
    # train_model()
    
    # Run interactive predictor
    interactive_predictor()