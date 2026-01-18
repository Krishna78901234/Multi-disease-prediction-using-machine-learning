# Multi-Disease Prediction System

A comprehensive Machine Learning application capable of predicting **Heart Disease**, **Diabetes**, and **Malaria**. The system uses a user-friendly Graphical User Interface (GUI) built with **Tkinter** to accept user inputs and display predictions with visual analytics.

## 🚀 Features

* **Heart Disease Prediction:** Uses a Voting Classifier model to analyze patient vitals (Age, BP, Cholesterol, etc.).
* **Diabetes Prediction:** Analyzes health metrics (Glucose, BMI, Insulin, etc.) using a Voting Classifier.
* **Malaria Detection:** Utilizes a Deep Learning model (**VGG19**) to classify cell images as "Parasitized" or "Uninfected".
* **Data Visualization:**
    * Pie charts for prediction confidence/results.
    * Histograms for dataset distribution analysis (Age, Glucose levels).

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **GUI:** Tkinter
* **Machine Learning:** Scikit-learn (Voting Classifier), Joblib
* **Deep Learning:** TensorFlow / Keras (VGG19)
* **Data Processing:** NumPy
* **Visualization:** Matplotlib

## 📂 Project Structure

```text
Multi-disease-prediction-using-machine-learning/
│
├── main.py                # The main application script (GUI logic)
├── model/                 # Directory containing trained models and data
│   ├── voting_classifier_model.pkl  # Diabetes Model
│   ├── voting_classifier_heart.pkl  # Heart Disease Model
│   ├── modelvgg19.h5                # Malaria Deep Learning Model (Large File)
│   ├── scaler.pkl                   # Data Scaler
│   ├── heart_data.npy               # Numpy array for heart data viz
│   └── diabetes_data.npy            # Numpy array for diabetes data viz
│
├── README.md              # Project documentation
└── requirements.txt       # List of dependencies
```


# Clone the repository
git clone [https://github.com/Pravanjan78901234/Multi-disease-prediction-using-machine-learning.git](https://github.com/Krishna78901234/Multi-disease-prediction-using-machine-learning.git)

# Navigate into the folder
cd Multi-disease-prediction-using-machine-learning

# Pull the large model file
git lfs pull

# Install dependencies
1. Make sure you have Python installed.
2. Run:pip install numpy joblib matplotlib scikit-learn tensorflow
3. Note: Tkinter usually comes pre-installed with Python. If you get an error, you may need to install python-tk.

# 🖥️ How to Run
1. Open your terminal or command prompt in the project directory.
2. Run the main script:python main.py
3. The GUI window will appear. Select the disease you want to predict from the menu.

## 🧠 Model Details

| Disease | Model Type | Input Type |
| :--- | :--- | :--- |
| **Heart Disease** | Voting Classifier (Ensemble) | Numerical Data (Age, CP, Thalach, etc.) |
| **Diabetes** | Voting Classifier (Ensemble) | Numerical Data (Glucose, BMI, Insulin, etc.) |
| **Malaria** | VGG19 (CNN - Deep Learning) | Image Data (.jpg, .png) |

# ⚠️ Important Note on Large Files
1. The file model/modelvgg19.h5 exceeds 100MB.
2. It is stored using Git Large File Storage (LFS).
3. If you download the zip file or clone without LFS, the model file may be a small pointer file (1KB) instead of the full model.
4. Ensure you run git lfs pull to get the actual model.



