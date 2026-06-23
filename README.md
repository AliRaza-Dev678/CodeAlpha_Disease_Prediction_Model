# ❤️ Heart Disease Prediction Model

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Logistic%20Regression-Classification-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
  <img src="https://img.shields.io/badge/CodeAlpha-Internship-red?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

<p align="center">
  A machine learning project that predicts the <strong>presence or absence of heart disease</strong> in a patient based on clinical diagnostic features, built as part of the <strong>CodeAlpha Data Science Internship</strong>. The project includes both a <strong>prediction model</strong> and an interactive <strong>Jupyter-based web application</strong>.
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Model Details](#-model-details)
- [Interactive App](#-interactive-app)
- [Results](#-results)
- [Technologies Used](#-technologies-used)
- [Getting Started](#-getting-started)
- [How to Run](#-how-to-run)
- [Key Concepts Covered](#-key-concepts-covered)
- [Medical Disclaimer](#️-medical-disclaimer)
- [Author](#-author)

---

## 🧠 Overview

This project applies **supervised machine learning** to predict whether a patient has heart disease based on 13 clinical attributes. It is built on the **Cleveland Heart Disease Dataset** — one of the most widely used medical datasets in machine learning research.

The repository includes two notebooks:
- **`Disease_Prediction.ipynb`** — the core ML model: EDA, preprocessing, training, and evaluation
- **`heart_app.ipynb`** — an interactive Jupyter application where users can input patient data and receive a real-time heart disease prediction

This project was developed as part of the **CodeAlpha Data Science Internship** program.

---

## 🎯 Problem Statement

Cardiovascular diseases are the **leading cause of death globally**, responsible for an estimated 17.9 million lives each year according to the WHO. Early and accurate detection is the most powerful tool to reduce this burden.

This project trains a machine learning classifier that predicts the likelihood of heart disease from routine clinical measurements — enabling faster screening, reducing diagnostic costs, and supporting clinicians in early intervention decisions.

---

## 📊 Dataset

**`heart.csv`** — Cleveland Heart Disease Dataset

| Property          | Details                                         |
|-------------------|-------------------------------------------------|
| Total Samples     | 303 patient records                             |
| Input Features    | 13 clinical attributes                          |
| Target Variable   | `target` — 0 = No Heart Disease, 1 = Heart Disease |
| Task Type         | Binary Classification                           |
| Missing Values    | ✅ None — fully clean dataset                   |
| Source            | UCI Machine Learning Repository (Cleveland DB)  |

### Feature Description

| Feature      | Full Name                      | Type        | Description                                                               |
|--------------|--------------------------------|-------------|---------------------------------------------------------------------------|
| `age`        | Age                            | Numerical   | Patient's age in years                                                    |
| `sex`        | Sex                            | Categorical | 1 = Male, 0 = Female                                                      |
| `cp`         | Chest Pain Type                | Categorical | 0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal, 3 = Asymptomatic |
| `trestbps`   | Resting Blood Pressure         | Numerical   | Resting blood pressure in mm Hg on admission                              |
| `chol`       | Serum Cholesterol              | Numerical   | Serum cholesterol in mg/dL                                                |
| `fbs`        | Fasting Blood Sugar            | Categorical | 1 = Fasting blood sugar > 120 mg/dL, 0 = otherwise                       |
| `restecg`    | Resting ECG Results            | Categorical | 0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy  |
| `thalach`    | Max Heart Rate Achieved        | Numerical   | Maximum heart rate achieved during exercise                               |
| `exang`      | Exercise Induced Angina        | Categorical | 1 = Yes, 0 = No                                                           |
| `oldpeak`    | ST Depression                  | Numerical   | ST depression induced by exercise relative to rest                        |
| `slope`      | Slope of Peak Exercise ST      | Categorical | 0 = Upsloping, 1 = Flat, 2 = Downsloping                                 |
| `ca`         | Major Vessels Colored          | Numerical   | Number of major vessels (0–3) colored by fluoroscopy                     |
| `thal`       | Thalassemia                    | Categorical | 1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect                      |
| `target`     | **Heart Disease (Target)**     | Binary      | **1 = Heart Disease Present, 0 = No Heart Disease**                      |

---

## 📁 Project Structure

```
CodeAlpha_Disease_Prediction_Model/
│
├── Disease_Prediction.ipynb         ← Core ML model notebook
│   ├── 1. Import Libraries
│   ├── 2. Load & Explore Dataset (heart.csv)
│   ├── 3. Exploratory Data Analysis (EDA)
│   │     ├── Target Class Distribution
│   │     ├── Correlation Heatmap
│   │     ├── Feature vs Target Analysis
│   │     └── Age & Chest Pain Distributions
│   ├── 4. Data Preprocessing & Feature-Target Split
│   ├── 5. Train-Test Split
│   ├── 6. Train Logistic Regression Model
│   ├── 7. Evaluate Model (Accuracy, Confusion Matrix, Report)
│   └── 8. Predictions on New Patient Data
│
├── heart_app.ipynb                  ← Interactive Jupyter prediction app
│   ├── Load Trained Model
│   ├── Accept Patient Input (13 clinical features)
│   └── Output: Heart Disease Prediction Result
│
├── heart.csv                        ← Cleveland Heart Disease Dataset
└── README.md
```

---

## 🤖 Model Details

### Algorithm: Logistic Regression

**Logistic Regression** is a well-established binary classification algorithm that models the probability of a class using the sigmoid function. It is widely used in medical diagnostics due to its:

- 📖 **Interpretability** — coefficients reveal which features increase/decrease disease risk
- ✅ **Reliability** — strong baseline performance on clinical datasets
- ⚡ **Efficiency** — fast training and inference on small medical datasets
- 🏥 **Clinical Trust** — transparent decision-making suitable for healthcare applications

**Training Configuration:**

| Parameter          | Value                             |
|--------------------|-----------------------------------|
| Algorithm          | Logistic Regression               |
| Solver             | lbfgs                             |
| Max Iterations     | 1000                              |
| Evaluation Metrics | Accuracy, Precision, Recall, F1   |
| Train-Test Split   | 80% Train / 20% Test              |
| Random State       | 2                                 |

### Preprocessing Steps

| Step                        | Details                                                      |
|-----------------------------|--------------------------------------------------------------|
| Data Loading                | Loaded from `heart.csv` via Pandas                           |
| Missing Value Check         | No missing values — no imputation required                   |
| Feature-Target Separation   | `X` = 13 features, `y` = `target` column                    |
| Feature Scaling             | StandardScaler applied to normalize numerical features       |
| Train-Test Split            | 80/20 stratified split to maintain class balance             |

---

## 💻 Interactive App

The **`heart_app.ipynb`** notebook serves as an interactive **patient prediction tool**. It allows users to:

1. **Enter patient clinical values** — age, sex, chest pain type, blood pressure, cholesterol, ECG results, and more
2. **Run the trained model** — applies the same preprocessing pipeline used during training
3. **Receive an instant prediction** — outputs whether the patient is at risk of heart disease or not

This demonstrates how a trained ML model can be packaged into a simple, user-facing healthcare tool.

---

## 📈 Results

| Metric              | Score      |
|---------------------|------------|
| Training Accuracy   | ~85–88%    |
| Test Accuracy       | ~82–85%    |
| Precision           | ~84%       |
| Recall              | ~85%       |
| F1-Score            | ~84%       |

> **Recall is the most critical metric** in heart disease detection — a high recall ensures that patients with heart disease are rarely missed (minimizing dangerous false negatives).

### Key Predictive Indicators

Based on the model's learned coefficients, the most influential features for heart disease prediction are:

| Feature         | Association with Heart Disease                                  |
|-----------------|-----------------------------------------------------------------|
| `cp`            | Asymptomatic chest pain strongly associated with disease        |
| `thalach`       | Lower max heart rate during exercise correlates with disease    |
| `ca`            | More vessels colored by fluoroscopy = higher disease risk       |
| `thal`          | Reversible defect in thalassemia = higher risk                  |
| `oldpeak`       | Higher ST depression = greater disease likelihood               |
| `exang`         | Exercise-induced angina is a strong disease indicator           |

---

## 🛠️ Technologies Used

| Technology       | Purpose                                         |
|------------------|-------------------------------------------------|
| Python 3.8+      | Programming Language                            |
| Scikit-Learn     | Logistic Regression, StandardScaler, Metrics    |
| Pandas           | Data Loading, Exploration & Manipulation        |
| NumPy            | Numerical Computations                          |
| Matplotlib       | Training Visualizations & EDA Plots             |
| Seaborn          | Correlation Heatmap & Feature Distribution Plots|
| Jupyter Notebook | Model Development & Interactive App Environment |

---

## 🚀 Getting Started

### Prerequisites

Make sure Python 3.8+ is installed, then install the required libraries:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn jupyter
```

### Clone the Repository

```bash
git clone https://github.com/AliRaza-Dev678/CodeAlpha_Disease_Prediction_Model.git
cd CodeAlpha_Disease_Prediction_Model
```

---

## ▶️ How to Run

### Run the ML Model
1. Launch Jupyter Notebook:
```bash
jupyter notebook
```
2. Open and run all cells in:
```
Disease_Prediction.ipynb
```

### Run the Interactive App
1. Open and run all cells in:
```
heart_app.ipynb
```
2. Enter patient clinical values in the input cells and get a real-time prediction.

> Make sure `heart.csv` is in the **same folder** as both notebooks before running.

---

## 📚 Key Concepts Covered

- ✅ **Medical EDA** — Analyzing class balance, feature distributions, age & gender patterns
- ✅ **Correlation Heatmap** — Understanding relationships between 13 clinical features
- ✅ **Feature Scaling** — Applying StandardScaler for logistic regression performance
- ✅ **Binary Classification** — Sigmoid output for Heart Disease (1) vs No Disease (0)
- ✅ **Precision vs Recall Trade-off** — Understanding why recall matters more in disease detection
- ✅ **Confusion Matrix Analysis** — Visualizing true/false positives and negatives
- ✅ **Classification Report** — Full per-class precision, recall, and F1-score breakdown
- ✅ **Interactive Prediction App** — Deploying trained model as a usable patient screening tool
- ✅ **CodeAlpha Internship Project** — Real-world applied ML in healthcare domain

---

## ⚠️ Medical Disclaimer

> This project is built for **educational and research purposes only** as part of the CodeAlpha Data Science Internship. The model is **not intended for actual clinical diagnosis, medical screening, or healthcare decision-making**. Always consult a qualified cardiologist or medical professional for cardiac evaluation and diagnosis.

---

## 👨‍💻 Author

**Ali Raza**
*CodeAlpha Machine Learning Intern*

- GitHub: [@AliRaza-Dev678](https://github.com/AliRaza-Dev678)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute it.

---

<p align="center">
  ⭐ If you found this project helpful, please give it a star! ⭐
</p>
