import joblib
from main import preprocess_text

# Load the model and vectorizer
def load_model_and_vectorizer(model_path="spam_classifier_model.pkl", vectorizer_path="vectorizer.pkl"):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("Model and vectorizer loaded successfully")
    return model, vectorizer


# Classify a single input row
def classify_message(message, model, vectorizer):
    preprocessed_message = preprocess_text(message)
    transformed_message = vectorizer.transform([preprocessed_message])
    # print(transformed_message)x
    prediction = model.predict(transformed_message) # 0 or 1 Spam and 0 for Ham
    return "Spam" if prediction[0] == 1 else "Ham"


# Example of loading the model and vectorizer for application use
loaded_model, loaded_vectorizer = load_model_and_vectorizer()

# Example single message classification
message="hey lady lady how are you. You must looking for discounts this Christmas"
result = classify_message(message, loaded_model, loaded_vectorizer)
print(f"The message '{message}' is classified as: {result}")
