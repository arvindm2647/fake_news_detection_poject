#  Fake News Detection using Machine Learning

A machine learning project that detects whether a given news article is **real** or **fake** based on its content. This project leverages natural language processing (NLP) techniques and classification algorithms to classify news articles effectively.

---

##  Features

- Text preprocessing and vectorization using TF-IDF
- Binary classification using models like Logistic Regression, Naive Bayes, or Random Forest
- Training and evaluation of the model
- Streamlit web app interface for real-time news prediction

---

##  Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
```

### 2. Install dependencies

Make sure you have Python 3.7+ installed. Then install the required libraries:

```bash
pip install -r requirements.txt
```

### 3. Dataset

Use the [Kaggle Fake News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset) or any dataset containing `title`, `text`, and `label` columns.

Place your CSV dataset in the root folder as `news.csv`.

---

##  Model Training

To train the model:

```bash
python train_model.py
```

This script will:
- Load and clean the data
- Split into training and test sets
- Vectorize text using TF-IDF
- Train a Logistic Regression classifier
- Save the model and vectorizer as `.pkl` files

---

## Model Evaluation

After training, you can evaluate the model using:

```bash
python evaluate_model.py
```

This script will output:
- Accuracy
- Precision, Recall, F1-Score
- Confusion matrix

---

##  Web App (Optional)

To run the web app using Streamlit:

```bash
streamlit run app.py
```

Upload or enter the news text, and it will predict whether it's **Fake** or **Real**.

---

##  Project Structure

```
fake-news-detection/
│
├── data/
│   └── news.csv                 # Dataset
├── models/
│   ├── model.pkl                # Trained model
│   └── vectorizer.pkl           # TF-IDF Vectorizer
├── train_model.py               # Training script
├── evaluate_model.py            # Evaluation script
├── app.py                       # Streamlit web app
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

##  Technologies Used

- Python
- Scikit-learn
- Pandas & NumPy
- NLTK / SpaCy
- Streamlit (optional for UI)
- TF-IDF Vectorizer

---

##  Example

```
Input:
"Donald Trump says vaccines cause autism in children"

Prediction:
 FAKE
```

---

##  Future Improvements

- Use deep learning (LSTM/BERT)
- Improve UI and mobile support
- Add data visualization
- Integrate with a browser extension

---

##  Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

##  License

This project is licensed under the [MIT License](LICENSE).

---

##  Acknowledgements

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Kaggle Datasets](https://www.kaggle.com/)
- [Streamlit](https://streamlit.io/)
