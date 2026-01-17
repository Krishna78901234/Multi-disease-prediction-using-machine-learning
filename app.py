import os
import numpy as np
import joblib
import tkinter as tk
from tkinter import messagebox, filedialog
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Get the absolute path dynamically
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths for models
diabetes_model_path = os.path.join(base_dir, "model", "voting_classifier_model.pkl")
heart_model_path = os.path.join(base_dir, "model", "voting_classifier_heart.pkl")
malaria_model_path = os.path.join(base_dir, "model", "modelvgg19.h5")

# Load models
diabetes_model = joblib.load(diabetes_model_path)
heart_model = joblib.load(heart_model_path)
malaria_model = load_model(malaria_model_path)

# Load the scaler (if available)
scaler_path = os.path.join(base_dir, "model", "scaler.pkl")
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

def visualize_data_distribution():
    """Visualizes the data distribution of Heart Disease and Diabetes datasets."""
    try:
        heart_data_path = os.path.join(base_dir, "model", "heart_data.npy")
        diabetes_data_path = os.path.join(base_dir, "model", "diabetes_data.npy")
        
        if os.path.exists(heart_data_path):
            heart_data = np.load(heart_data_path)
            plt.figure(figsize=(8, 5))
            plt.hist(heart_data[:, 0], bins=20, color='skyblue', edgecolor='black')  # Age distribution
            plt.title('Heart Disease Dataset Distribution (Age)')
            plt.xlabel('Age')
            plt.ylabel('Frequency')
            plt.show()

        if os.path.exists(diabetes_data_path):
            diabetes_data = np.load(diabetes_data_path)
            plt.figure(figsize=(8, 5))
            plt.hist(diabetes_data[:, 1], bins=20, color='lightgreen', edgecolor='black')  # Glucose distribution
            plt.title('Diabetes Dataset Distribution (Glucose Level)')
            plt.xlabel('Glucose Level')
            plt.ylabel('Frequency')
            plt.show()
            
    except Exception as e:
        messagebox.showerror("Error", str(e))

def show_heart_form():
    """Displays input form for Heart Disease prediction."""
    form = tk.Toplevel(root)
    form.title("Heart Disease Prediction")

    labels = [
        "Age:", "Sex (0 = Female, 1 = Male):", "Chest Pain Type (0-3):",
        "Resting Blood Pressure:", "Cholesterol:", "Fasting Blood Sugar (1 = True, 0 = False):",
        "Resting ECG (0-2):", "Max Heart Rate:", "Exercise Induced Angina (1 = Yes, 0 = No):",
        "Oldpeak:", "Slope (0-2):", "Number of Major Vessels (0-3):", "Thalassemia (1-3):"
    ]

    entry_widgets = {}

    for i, label in enumerate(labels):
        tk.Label(form, text=label).grid(row=i, column=0)
        entry = tk.Entry(form)
        entry.grid(row=i, column=1)
        entry_widgets[label] = entry  

    def predict_heart():
        try:
            data = np.array([float(entry_widgets[label].get()) for label in labels]).reshape(1, -1)
            if scaler:
                data = scaler.transform(data)
            result = heart_model.predict(data)
            prediction = 'Positive' if result[0] == 1 else 'Negative'

            plt.figure(figsize=(5, 5))
            plt.pie([1, 0], labels=[prediction, 'Negative'], colors=['#ff9999','#66b3ff'], autopct='%1.1f%%', startangle=90)
            plt.title('Heart Disease Prediction')
            plt.axis('equal')
            plt.show()

            messagebox.showinfo("Prediction Result", f"Heart Disease Prediction: {prediction}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    tk.Button(form, text="Predict", command=predict_heart).grid(row=len(labels), column=1)

def show_diabetes_form():
    """Displays input form for Diabetes prediction."""
    form = tk.Toplevel(root)
    form.title("Diabetes Prediction")

    labels = [
        "Pregnancies:", "Glucose:", "Blood Pressure:", "Skin Thickness:", 
        "Insulin:", "BMI:", "Diabetes Pedigree Function:", "Age:"
    ]

    entry_widgets = {}

    for i, label in enumerate(labels):
        tk.Label(form, text=label).grid(row=i, column=0)
        entry = tk.Entry(form)
        entry.grid(row=i, column=1)
        entry_widgets[label] = entry  

    def predict_diabetes():
        try:
            data = np.array([float(entry_widgets[label].get()) for label in labels]).reshape(1, -1)
            if scaler:
                data = scaler.transform(data)
            result = diabetes_model.predict(data)
            prediction = 'Positive' if result[0] == 1 else 'Negative'

            plt.figure(figsize=(5, 5))
            plt.pie([1, 0], labels=[prediction, 'Negative'], colors=['#ff9999','#66b3ff'], autopct='%1.1f%%', startangle=90)
            plt.title('Diabetes Prediction')
            plt.axis('equal')
            plt.show()

            messagebox.showinfo("Prediction Result", f"Diabetes Prediction: {prediction}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    tk.Button(form, text="Predict", command=predict_diabetes).grid(row=len(labels), column=1)

def predict_malaria():
    """Predicts Malaria using an image classifier."""
    try:
        file_path = filedialog.askopenfilename(title="Select Malaria Image", filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if not file_path:
            return
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        result = malaria_model.predict(img_array)
        prediction = 'Malaria' if result[0][0] > 0.5 else 'No Malaria'

        plt.figure(figsize=(5, 5))
        plt.pie([1, 0], labels=[prediction, 'No Malaria'], colors=['#ff9999','#66b3ff'], autopct='%1.1f%%', startangle=90)
        plt.title('Malaria Prediction')
        plt.axis('equal')
        plt.show()

        messagebox.showinfo("Prediction Result", f"Malaria Prediction: {prediction}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Main Application Window
root = tk.Tk()
root.title("Multi-Disease Prediction System")

tk.Button(root, text="Predict Heart Disease", command=show_heart_form).pack()
tk.Button(root, text="Predict Diabetes", command=show_diabetes_form).pack()
tk.Button(root, text="Predict Malaria", command=predict_malaria).pack()
tk.Button(root, text="Visualize Data Distribution", command=visualize_data_distribution).pack()

root.mainloop()
