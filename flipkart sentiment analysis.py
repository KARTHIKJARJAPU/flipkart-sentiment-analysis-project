import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob  # Import TextBlob for sentiment analysis

df = pd.read_csv('flipkart.csv')
df_subset = df.head(100)

# Perform sentiment analysis using TextBlob
df_subset['Sentiment'] = df_subset['Review'].apply(lambda review: TextBlob(review).sentiment.polarity) 
# polarity ranges from -1 (negative) to 1 (positive)

# Convert sentiment polarity to categorical labels (Positive/Negative)
df_subset['Sentiment_Label'] = df_subset['Sentiment'].apply(lambda polarity: 'Positive' if polarity > 0 else 'Negative')

# Scatter plot of sentiment for the subset
plt.figure(figsize=(8, 6))
plt.scatter(range(len(df_subset)), df_subset['Sentiment_Label'], 
            c=['green' if sentiment == 'Positive' else 'red' for sentiment in df_subset['Sentiment_Label']], alpha=0.5)
plt.xlabel("Review Index")
plt.ylabel("Sentiment (Green: Positive, Red: Negative)")
plt.title("Sentiment Analysis of Customer Reviews (First 20 Reviews)")
plt.show()