import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load heart disease dataset
data = pd.read_csv('C:\\Users\\prava\\Desktop\\multi disease prediction\\multi disease prediction\\dataset\\heart.csv')

# Features and target variable
X = data.drop('target', axis=1)  # Assuming 'target' is the label column
y = data['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
ab_classifier = AdaBoostClassifier(n_estimators=100, random_state=42, algorithm='SAMME')

# Create Voting Classifier
voting_classifier = VotingClassifier(
    estimators=[('rf', rf_classifier), ('gb', gb_classifier), ('ab', ab_classifier)],
    voting='hard'
)

# Train and evaluate
voting_classifier.fit(X_train, y_train)
y_pred = voting_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Heart Disease Voting Classifier Accuracy: {accuracy:.2f}')

# Save model
joblib.dump(voting_classifier, 'C:\\Users\\prava\\Desktop\\multi disease prediction\\multi disease prediction\\model\\voting_classifier_heart.pkl')
print("Heart Disease model saved successfully!")
