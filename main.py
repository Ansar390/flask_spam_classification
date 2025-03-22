import pandas as pd
import nltk
import string, joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
# nltk.download('stopwords')
# nltk.download('wordnet')


data= pd.read_csv("/Users/apple/Documents/nlp_ml_app/nlp_ml_flask_app/spam.csv",encoding="ISO-8859-1")
df = data[['v1', 'v2']]
df.columns = ['label', 'message']



def preprocess_text(text):
  text = text.lower()
  # print("Text: ",text)
  text = ''.join([char for char in text if char not in string.punctuation])
  # print("after punication removal: ",text)
  text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
  # print("after stopword removal: ",text)

  # Apply stemming
  # stemmer = PorterStemmer()
  # text = ' '.join([stemmer.stem(word) for word in text.split()])

  lemmatizer = WordNetLemmatizer()
  text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

  return text

df['cleaned_message'] = df['message'].apply(preprocess_text)

# print(df['cleaned_message'])


## Applying Feature vector 
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_message'])
# X.vocabulary_
# print(vectorizer.get_feature_names_out())
y = df['label'].map({'ham': 0, 'spam': 1})
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))



# Save the trained model and vectorizer
def save_model_and_vectorizer(model, vectorizer, model_path="spam_classifier_model.pkl", vectorizer_path="vectorizer.pkl"):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")


save_model_and_vectorizer(model, vectorizer)