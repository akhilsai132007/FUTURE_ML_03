import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def main():
    # 1. Load data
    print("Loading data...")
    df = pd.read_csv('data/resumes.csv')

    # 2. Clean missing values
    print("Cleaning missing values...")
    df.dropna(subset=['resume_text', 'skills', 'job_role'], inplace=True)

    # Combine resume_text and skills to create a richer feature
    df['combined_text'] = df['resume_text'] + " " + df['skills']

    # 3. Use TF-IDF Vectorizer
    print("Vectorizing text data...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(df['combined_text'])
    y = df['job_role']

    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train models
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)

    print("Training Multinomial Naive Bayes...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    nb_acc = accuracy_score(y_test, nb_pred)

    # 5. Compare accuracy
    print("\n--- Model Comparison ---")
    print(f"Logistic Regression Accuracy: {lr_acc * 100:.2f}%")
    print(f"Multinomial Naive Bayes Accuracy: {nb_acc * 100:.2f}%")

    # Select the best model
    if lr_acc >= nb_acc:
        best_model = lr_model
        best_pred = lr_pred
        best_name = "Logistic Regression"
    else:
        best_model = nb_model
        best_pred = nb_pred
        best_name = "Multinomial Naive Bayes"

    print(f"\nBest model selected: {best_name}")

    # 6. Print classification report
    print("\n--- Classification Report ---")
    print(classification_report(y_test, best_pred))

    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)

    # 7. Save best model to models/resume_model.pkl
    print("Saving the best model...")
    with open('models/resume_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    # 8. Save vectorizer to models/vectorizer.pkl
    print("Saving the vectorizer...")
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    print("Successfully saved model and vectorizer to 'models/' folder!")

if __name__ == "__main__":
    main()
