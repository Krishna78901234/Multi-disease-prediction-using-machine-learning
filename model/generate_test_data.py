import numpy as np
import os

# Get the base directory dynamically
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, "model")

# Ensure the 'model' directory exists
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Define paths for heart and diabetes test files
test_heart_path = os.path.join(model_dir, "test_heart.npy")
test_heart_labels_path = os.path.join(model_dir, "test_heart_labels.npy")

test_diabetes_path = os.path.join(model_dir, "test_diabetes.npy")
test_diabetes_labels_path = os.path.join(model_dir, "test_diabetes_labels.npy")

# Create dummy test data
num_samples = 10  # Number of test samples

# For Heart Disease (13 features)
X_test_heart = np.random.rand(num_samples, 13)  # 13 input features
y_test_heart = np.random.randint(0, 2, num_samples)  # Binary labels (0 or 1)

# For Diabetes (8 features)
X_test_diabetes = np.random.rand(num_samples, 8)  # 8 input features
y_test_diabetes = np.random.randint(0, 2, num_samples)  # Binary labels (0 or 1)

# Save test data
np.save(test_heart_path, X_test_heart)
np.save(test_heart_labels_path, y_test_heart)

np.save(test_diabetes_path, X_test_diabetes)
np.save(test_diabetes_labels_path, y_test_diabetes)

print("✅ Test files for Heart Disease and Diabetes created successfully!")
