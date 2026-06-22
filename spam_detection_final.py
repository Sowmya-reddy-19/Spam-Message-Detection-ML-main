import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")

# Keep required columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Convert labels to numbers
y = data['label'].map({'ham': 0, 'spam': 1})

# Convert text into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message'])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Algorithm 1: Naive Bayes
# ---------------------------
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

nb_predictions = nb_model.predict(X_test)

print("\n===== NAIVE BAYES =====")
print("Accuracy:", accuracy_score(y_test, nb_predictions))

print("\nClassification Report:")
print(classification_report(y_test, nb_predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, nb_predictions))

# ROC-AUC
nb_prob = nb_model.predict_proba(X_test)[:, 1]
print("\nROC-AUC Score:",
      roc_auc_score(y_test, nb_prob))

# Cross Validation
cv_scores = cross_val_score(nb_model, X, y, cv=5)

print("\nCross Validation Scores:")
print(cv_scores)

print("Average CV Score:",
      cv_scores.mean())

# ---------------------------
# Algorithm 2: Logistic Regression
# ---------------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

print("\n===== LOGISTIC REGRESSION =====")
print("Accuracy:",
      accuracy_score(y_test, lr_predictions))

# ---------------------------
# Compare Algorithms
# ---------------------------
print("\n===== ALGORITHM COMPARISON =====")
print("Naive Bayes Accuracy:",
      accuracy_score(y_test, nb_predictions))

print("Logistic Regression Accuracy:",
      accuracy_score(y_test, lr_predictions))

# ---------------------------
# User Message Prediction
# ---------------------------
message = input("\nEnter a message: ")

message_vector = vectorizer.transform([message])

prediction = nb_model.predict(message_vector)

if prediction[0] == 1:
    print("Result: Spam Message")
else:
    print("Result: Not Spam (Ham) Message")

# ---------------------------
# Pie Chart
# ---------------------------
spam_count = data['label'].value_counts()

plt.figure(figsize=(5, 5))
plt.pie(
    spam_count,
    labels=spam_count.index,
    autopct='%1.1f%%'
)
plt.title("Spam vs Ham Messages")
plt.show()