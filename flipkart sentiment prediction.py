
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from textblob import TextBlob

df = pd.read_csv('flipkart.csv')
df['Sentiment_Label'] = df['Review'].apply(lambda review: 'Positive' if TextBlob(review).sentiment.polarity > 0 else 'Negative')

X_train, X_test, y_train, y_test = train_test_split(
    df['Review'], df['Sentiment_Label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
def predict_sentiment(review):
    review_vec = vectorizer.transform([review])
    predicted_sentiment = model.predict(review_vec)[0]
    return predicted_sentiment
while True:
    user_input = input("Enter a review (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break

    sentiment = predict_sentiment(user_input)
    print(f"Predicted sentiment: {sentiment}")