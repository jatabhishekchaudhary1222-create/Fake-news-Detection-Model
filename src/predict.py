import re
import string
import joblib
def clean_text(text):
    text = text.lower()
    # Remove Puntuation(Using str.transalte with str.maketrans())
    text.translate(str.maketrans('', '', string.punctuation))
    # Remove Whitespaces(re.sub{Regex Substitute})
    re.sub(r'\s+', ' ', text).strip()
    return text
model = joblib.load('models/fake_news_model.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')
def predict_news(text):
    cleaned = clean_text(text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return prediction
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
if __name__ == "__main__":
    sample = "Scientists confirm the moon landing was staged in a Hollywood studio"
    print(predict_news(sample))