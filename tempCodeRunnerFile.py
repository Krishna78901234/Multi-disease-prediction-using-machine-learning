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

# Global variables for input fields
entry_age = entry_sex = entry_cp = entry_trestbps = entry_chol = entry_fbs = None
entry_restecg = entry_thalach = entry_exang = entry_oldpeak = entry_slope = entry_ca = entry_thal = None

entry_pregnancies = entry_glucose = entry_blood_pressure = entry_skin_thickness = None
entry_insulin = entry_bmi = entry_dpf = entry_age_diabetes = None

def predict_heart():
    try:
        data = np.array([
            int(entry_age.get()), int(entry_sex.get()), int(entry_cp.get()),
            int(entry_trestbps.get()), int(entry_chol.get()), int(entry_fbs.get()),
            int(entry_restecg.get()), int(entry_thalach.get()), int(entry_exang.get()),
            float(entry_oldpeak.get()), int(entry_slope.get()), int(entry_ca.get()),
            int(entry_thal.get())
        ]).reshape(1, -1)

        if scaler:
            data = scaler.transform(data)

        result = heart_model.predict(data)
        prediction = 'Positive' if result[0] == 1 else 'Negative'
        messagebox.showinfo("Prediction Result", f"Heart Disease Prediction: {prediction}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def predict_diabetes():
    try:
        data = np.array([
            int(entry_pregnancies.get()), int(entry_glucose.get()), int(entry_blood_pressure.get()),
            int(entry_skin_thickness.get()), int(entry_insulin.get()), float(entry_bmi.get()),
            float(entry_dpf.get()), int(entry_age_diabetes.get())
        ]).reshape(1, -1)

        if scaler:
            data = scaler.transform(data)

        result = diabetes_model.predict(data)
        prediction = 'Positive' if result[0] == 1 else 'Negative'
        messagebox.showinfo("Prediction Result", f"Diabetes Prediction: {prediction}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def predict_malaria():
    try:
        file_path = filedialog.askopenfilename(title="Select Malaria Image", filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if not file_path:
            return
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        result = malaria_model.predict(img_array)
        prediction = 'Malaria' if result[0][0] > 0.5 else 'No Malaria'
        messagebox.showinfo("Prediction Result", f"Malaria Prediction: {prediction}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def show_heart_form():
    global entry_age, entry_sex, entry_cp, entry_trestbps, entry_chol, entry_fbs
    global entry_restecg, entry_thalach, entry_exang, entry_oldpeak, entry_slope, entry_ca, entry_thal

    form_frame = tk.Toplevel(root)
    form_frame.title("Heart Disease Prediction Form")

    labels = [
        "Age:", "Sex (0 = Female, 1 = Male):", "Chest Pain Type (0-3):",
        "Resting Blood Pressure:", "Cholesterol:", "Fasting Blood Sugar (1 = True, 0 = False):",
        "Resting ECG (0-2):", "Max Heart Rate:", "Exercise Induced Angina (1 = Yes, 0 = No):",
        "Oldpeak:", "Slope (0-2):", "Number of Major Vessels (0-3):", "Thalassemia (1-3):"
    ]

    entries = []
    for i, label in enumerate(labels):
        tk.Label(form_frame, text=label).grid(row=i, column=0)
        entry = tk.Entry(form_frame)
        entry.grid(row=i, column=1)
        entries.append(entry)

    (entry_age, entry_sex, entry_cp, entry_trestbps, entry_chol, entry_fbs,
     entry_restecg, entry_thalach, entry_exang, entry_oldpeak, entry_slope, entry_ca, entry_thal) = entries

    tk.Button(form_frame, text="Predict", command=predict_heart).grid(row=len(labels), column=1)

def show_diabetes_form():
    global entry_pregnancies, entry_glucose, entry_blood_pressure, entry_skin_thickness
    global entry_insulin, entry_bmi, entry_dpf, entry_age_diabetes

    form_frame = tk.Toplevel(root)
    form_frame.title("Diabetes Prediction Form")

    labels = [
        "Pregnancies:", "Glucose Level:", "Blood Pressure:", "Skin Thickness:",
        "Insulin Level:", "BMI:", "Diabetes Pedigree Function:", "Age:"
    ]

    entries = []
    for i, label in enumerate(labels):
        tk.Label(form_frame, text=label).grid(row=i, column=0)
        entry = tk.Entry(form_frame)
        entry.grid(row=i, column=1)
        entries.append(entry)

    (entry_pregnancies, entry_glucose, entry_blood_pressure, entry_skin_thickness,
     entry_insulin, entry_bmi, entry_dpf, entry_age_diabetes) = entries

    tk.Button(form_frame, text="Predict", command=predict_diabetes).grid(row=len(labels), column=1)

def show_accuracy():
    try:
        accuracies = {}

        # Load test data for Heart (if available)
        heart_data_path = os.path.join(base_dir, "model", "test_heart.npy")
        heart_labels_path = os.path.join(base_dir, "model", "test_heart_labels.npy")

        if os.path.exists(heart_data_path) and os.path.exists(heart_labels_path):
            X_test_heart = np.load(heart_data_path)
            y_true_heart = np.load(heart_labels_path)

            if scaler:
                X_test_heart = scaler.transform(X_test_heart)

            y_pred_heart = heart_model.predict(X_test_heart)
            accuracies['Heart'] = accuracy_score(y_true_heart, y_pred_heart)

        # Load test data for Diabetes (if available)
        diabetes_data_path = os.path.join(base_dir, "model", "test_diabetes.npy")
        diabetes_labels_path = os.path.join(base_dir, "model", "test_diabetes_labels.npy")

        if os.path.exists(diabetes_data_path) and os.path.exists(diabetes_labels_path):
            X_test_diabetes = np.load(diabetes_data_path)
            y_true_diabetes = np.load(diabetes_labels_path)

            if scaler:
                X_test_diabetes = scaler.transform(X_test_diabetes)

            y_pred_diabetes = diabetes_model.predict(X_test_diabetes)
            accuracies['Diabetes'] = accuracy_score(y_true_diabetes, y_pred_diabetes)

        # Load test data for Malaria (if available)
        malaria_data_path = os.path.join('C:\\Users\\prava\\Desktop\\multi disease prediction\\multi disease prediction\\dataset\\maleriadata\\Test')
        malaria_labels_path = os.path.join(base_dir, "model", "test_malaria_labels.npy")

        if os.path.exists(malaria_data_path) and os.path.exists(malaria_labels_path):
            X_test_malaria = np.load(malaria_data_path)
            y_true_malaria = np.load(malaria_labels_path)

            y_pred_malaria = malaria_model.predict(X_test_malaria)
            y_pred_malaria = (y_pred_malaria > 0.5).astype(int)

            accuracies['Malaria'] = accuracy_score(y_true_malaria, y_pred_malaria)

        # If no test data is found, show an error message
        if not accuracies:
            messagebox.showerror("Error", "No test data found for any disease!")
            return

        # Plot accuracies
        plt.figure(figsize=(6, 4))
        plt.bar(accuracies.keys(), accuracies.values(), color=['red', 'blue', 'green'])
        plt.ylim(0, 1)
        plt.title('Model Accuracies')
        plt.ylabel('Accuracy Score')
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", str(e))



root = tk.Tk()
root.title("Multi-Disease Prediction System")

tk.Button(root, text="Predict Heart Disease", command=show_heart_form).pack()
tk.Button(root, text="Predict Diabetes", command=show_diabetes_form).pack()
tk.Button(root, text="Predict Malaria", command=predict_malaria).pack()
tk.Button(root, text="Show Accuracy", command=show_accuracy).pack()

root.mainloop()


