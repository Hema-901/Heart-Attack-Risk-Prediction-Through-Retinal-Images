# Heart Attack Risk Prediction Through Retinal Images

## 📌 Project Overview

This project is an AI-powered web application that predicts cardiovascular risk levels using retinal fundus images. The system uses Deep Learning and Computer Vision techniques to analyze retinal images and classify them into **High Risk** or **Low Risk** categories.

The application is developed using **Python, TensorFlow/Keras, Flask, HTML, CSS, and JavaScript**.

---

## 🚀 Features

* Upload retinal fundus images
* AI-based cardiovascular risk prediction
* Deep Learning model integration
* Real-time prediction using Flask
* User-friendly web interface
* Transfer Learning using MobileNetV2

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* Flask
* MobileNetV2
* NumPy
* HTML
* CSS
* JavaScript

---

## 🧠 Deep Learning Model

The project uses **Transfer Learning** with **MobileNetV2** for retinal image classification.

### Workflow:

1. Image Upload
2. Image Preprocessing
3. Deep Learning Prediction
4. Risk Classification
5. Display Prediction Result

---

## 📂 Project Structure

```text
Heart Attack Prediction
│
├── app.py
├── train_model.py
├── requirements.txt
├── .gitignore
│
├── model
│   └── best_model.keras
│
├── retinal_images
│   ├── high_risk
│   └── low_risk
│
├── static
│
├── templates
│   └── index.html
```

---

## ▶️ How to Run the Project

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Hema-901/Heart-Attack-Risk-Prediction-Through-Retinal-Images.git
```

### 2️⃣ Navigate to Project Folder

```bash
cd "Heart Attack Prediction"
```

### 3️⃣ Create Virtual Environment

```bash
python -m venv myenv
```

### 4️⃣ Activate Virtual Environment

#### Windows

```bash
.\myenv\Scripts\Activate
```

### 5️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

### 6️⃣ Run Flask Application

```bash
python app.py
```

### 7️⃣ Open Browser

```text
http://127.0.0.1:5000
```

---

## 📊 Model Training

To train the model:

```bash
python train_model.py
```

The trained model will be saved inside:

```text
model/best_model.keras
```

---

## 📸 Screenshots

(Add your project screenshots here)

---

## 📌 Future Improvements

* Improve model accuracy with larger datasets
* Add multiple disease classifications
* Deploy application to cloud platforms
* Add advanced evaluation metrics
* Improve UI/UX design

---

## ⚠️ Disclaimer

This project is developed for educational and research purposes only. It is not intended for clinical or medical diagnosis.

---

## 👩‍💻 Author

Hema

GitHub:
https://github.com/Hema-901
