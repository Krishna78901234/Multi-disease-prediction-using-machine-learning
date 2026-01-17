import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load your dataset (replace with your actual dataset)
data = pd.read_csv('C:\\Users\\prava\\Desktop\\multi disease prediction\\multi disease prediction\\dataset\\data.csv')

# Features and target variable
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize individual classifiers
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
ab_classifier = AdaBoostClassifier(n_estimators=100, random_state=42,algorithm='SAMME')

# Create a Voting Classifier
voting_classifier = VotingClassifier(
    estimators=[
        ('rf', rf_classifier),
        ('gb', gb_classifier),
        ('ab', ab_classifier)
    ],
    voting='hard'  # Use 'soft' for probability-based voting
)

# Train the Voting Classifier
voting_classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = voting_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Voting Classifier Accuracy: {accuracy:.2f}')

# Save the trained Voting Classifier model
joblib.dump(voting_classifier, 'C:\\Users\\prava\\Desktop\\multi disease prediction\\multi disease prediction\\model\\voting_classifier_model.pkl')
print("Voting Classifier model saved successfully!")
