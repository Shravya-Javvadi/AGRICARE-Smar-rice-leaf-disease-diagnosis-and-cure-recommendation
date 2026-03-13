# 🌾 Rice Leaf Disease Detection and Severity Estimation using Deep Learning

This project implements an **AI-powered rice leaf disease detection system** that can identify diseases from rice leaf images, estimate the **severity of infection**, generate **Grad-CAM explainability maps**, and recommend **treatment with dosage**.

The system uses **transfer learning models** such as **MobileNetV2, ResNet50, EfficientNetB0, and Xception** to classify rice leaf diseases. Among these, **Xception achieved the highest accuracy (~98%) and is used as the final model for prediction**.

---

# 📁 Project Structure

```
MAJOR_PROJECT
│
├── Dataset
│   ├── train_images
│   ├── test_images
│   ├── train.csv
│   └── sample_submission.csv
│
├── ModelSavedFiles
│   ├── paddy_final_model
│   ├── rice_leaf_detector.h5
│   └── rice_vs_nonrice_model.h5
│
├── ModelTrainedFiles
│   ├── EfficientNet.ipynb
│   ├── leafdisease.ipynb
│   ├── ModelsTraining.ipynb
│   ├── riceVsnonrice.ipynb
│   └── SeverityEstimation.ipynb
│
├── app.py
├── requirements.txt
├── README.md
├── temp_agricare.jpg
└── temp.jpg
```

---

# 🚀 Features

This system performs **multiple intelligent tasks** for rice crop disease analysis.

### 1️⃣ Rice Leaf Detection
The system first verifies whether the uploaded image belongs to a **rice leaf** before performing disease classification.

This helps prevent incorrect predictions on unrelated images.

---

### 2️⃣ Disease Classification

The project uses **Transfer Learning** with pretrained CNN models.

The following models were trained and evaluated:

| Model | Epochs | Accuracy |
|------|------|------|
| MobileNetV2 | 5 | ~88% |
| ResNet50 | 10 | ~90% |
| EfficientNetB0 | 10 | ~97% |
| Xception | 10 | ~98% |

The **EfficientNetB0** was selected as the final model due to its superior accuracy and robustness.

---

### 3️⃣ Severity Estimation

After detecting the disease, the system estimates the **severity of infection** by calculating the percentage of infected leaf area using image processing techniques.

Severity levels are categorized as:

| Infection Percentage | Severity Level |
|----------------------|---------------|
| 0% – 10% | Mild |
| 10% – 30% | Moderate |
| > 30% | Severe |

---

### 4️⃣ Explainable AI (Grad-CAM)

To increase transparency and trust in the model predictions, **Grad-CAM (Gradient-weighted Class Activation Mapping)** is used.

Grad-CAM highlights the **important regions of the leaf image that influenced the model's prediction**.

Outputs include:
- Original leaf image
- Disease affected region
- Grad-CAM heatmap visualization

---

### 5️⃣ Treatment Recommendation

Based on the detected disease and severity level, the system recommends appropriate **chemical treatment and dosage**.

Example recommendations:

| Disease | Severity | Treatment | Dosage |
|-------|-------|-------|-------|
| Brown Spot | Severe | Propiconazole | 1 ml/L |
| Bacterial Leaf Blight | Moderate | Copper Oxychloride | 2.5 g/L |
| Bacterial Leaf Blight | Mild | Streptocycline | 0.5 g/10L |

---

# 🧠 Methodology

The system follows the pipeline shown below:

```
Input Image
      │
      ▼
Rice Leaf Verification
      │
      ▼
Disease Classification (Xception CNN)
      │
      ▼
Severity Estimation
      │
      ▼
Explainable AI (Grad-CAM)
      │
      ▼
Treatment Recommendation
```

---

# 🧪 Dataset

Dataset used:

**Paddy Doctor: Paddy Disease Classification Dataset**

Source:
```
https://www.kaggle.com/datasets/dasa7753912/new-paddy-doctor-paddy-disease-classification
```

The dataset contains images of rice leaves belonging to the following classes:

- Bacterial Leaf Blight
- Brown Spot
- False Smut
- Healthy
- Hispa
- Leaf Blast
- Leaf Scald
- Narrow Brown Spot
- Sheath Blight
- Tungro

---

# ⚙️ Installation

### 1️⃣ Clone the repository

```
git clone https://github.com/yourusername/rice-leaf-disease-detection.git
cd rice-leaf-disease-detection
```

---

### 2️⃣ Install required libraries

```
pip install -r requirements.txt
```

---

### 3️⃣ Download Dataset using Kaggle

```
pip install kaggle
kaggle datasets download -d dasa7753912/new-paddy-doctor-paddy-disease-classification
```

Unzip the dataset inside the project directory.

---

# ▶️ Running the Application

Run the application using:

```
python app.py
```

Upload a rice leaf image to get:

- Disease prediction
- Severity estimation
- Grad-CAM visualization
- Treatment recommendation

---

# 📊 Model Performance

| Model | Accuracy |
|------|------|
| MobileNetV2 | ~88% |
| ResNet50 | ~90% |
| EfficientNetB0 | ~97% |
| **Xception** | **~98%** |

The **Xception architecture demonstrated the best classification performance**, making it suitable for deployment.

---

# 🧰 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Seaborn
- Kaggle API
- Transfer Learning
- Explainable AI (Grad-CAM)

---

# 🌱 Applications

This project can be used in several agricultural applications:

- Smart farming systems
- Early crop disease detection
- Precision agriculture
- Automated crop monitoring
- Agricultural decision support systems

---

# 🔮 Future Improvements

Potential enhancements include:

- Mobile application deployment
- Drone-based crop monitoring
- Real-time field disease detection
- Integration with IoT agricultural sensors
- Automated pesticide recommendation system

---

# 👩‍💻 Authors

Major Project: **Rice Leaf Disease Detection and Severity Estimation using Deep Learning**

Developed as part of an academic project in **Artificial Intelligence / Machine Learning**.

---

# 📜 License

This project is intended for **academic and research purposes only**.

