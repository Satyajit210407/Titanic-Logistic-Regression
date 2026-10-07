# 🚢 Titanic Survival Prediction — Logistic Regression Analysis

## 📌 Project Overview

This project focuses on **Logistic Regression Analysis of Titanic passenger data** using Python and Scikit-learn.

The notebook applies **Univariate Logistic Regression** and **Multivariate Logistic Regression** to analyze the relationship between passenger information and **survival**.

The main purpose of this project is to understand how logistic regression can be applied to a real-world dataset, perform predictions, and interpret the results.

> **Note:** This is an educational and analytical project focused on understanding classification with logistic regression. It is not a production-ready Machine Learning model or deployed prediction system.

---

## 🎯 Objective

The main objective of this project is to:

* Load and inspect the Titanic dataset
* Handle missing values
* Encode categorical variables
* Apply Univariate Logistic Regression
* Apply Multivariate Logistic Regression
* Generate survival predictions
* Evaluate model performance
* Compare the two models and interpret the results

> **Note:** The target variable is `survived`, a binary outcome (`0` = did not survive, `1` = survived). Because the target is categorical, this is a **classification** task, so Logistic Regression is used instead of Linear Regression.

---

## 📊 Logistic Regression Analysis

### 1. Univariate Logistic Regression

Univariate Logistic Regression uses a single independent variable to predict the probability of survival.

In this project, **passenger fare** is used as the only feature.

```text
P(survived = 1) = 1 / (1 + e^-(β₀ + β₁ × fare))
```

The analysis helps understand how much a passenger's fare alone explains their chance of survival.

### 2. Multivariate Logistic Regression

Multivariate Logistic Regression uses multiple independent variables to predict survival.

```text
P(survived = 1) = 1 / (1 + e^-(β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ))
```

The features used are `pclass`, `age`, `fare`, `sibsp` and `sex`. This provides a broader understanding of which factors influenced a passenger's chance of survival.

---

## 📊 Dataset

The dataset is the **Titanic dataset**, loaded directly from the Seaborn library. It contains 891 passengers and 15 columns.

Some important features include:

* `survived`
* `pclass`
* `sex`
* `age`
* `sibsp`
* `parch`
* `fare`
* `embarked`
* `class`
* `who`
* `adult_male`
* `deck`
* `embark_town`
* `alive`
* `alone`

### Target Variable

```text
survived
```

`survived` is used as the target variable. In the dataset, **549 passengers did not survive** and **342 survived**.

---

## 🧹 Data Preparation

| Column | Issue | Handling |
|---|---|---|
| `age` | 177 missing values | Filled with the median |
| `embarked` | 2 missing values | Filled with the mode |
| `embark_town` | 2 missing values | Filled with the mode |
| `deck` | 688 of 891 values missing | Not used |
| `sex` | Categorical | Label-encoded (female = 0, male = 1) |

---

## 🛠️ Technologies & Libraries

* **Python**
* **Jupyter Notebook**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Inspection
   ↓
Missing Value Handling
   ↓
Categorical Encoding
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Univariate Logistic Regression
   ↓
Multivariate Logistic Regression
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Result Interpretation
```

---

## 📈 Model Evaluation

Both models use an **80/20 train-test split** with `random_state=42` and are evaluated using:

* **Accuracy Score**
* **Confusion Matrix**
* **Model Coefficients and Intercept**

### Results

| Model | Features | Accuracy |
|---|---|---|
| Univariate | `fare` | 65.4% |
| Multivariate | `pclass`, `age`, `fare`, `sibsp`, `sex` | 80.4% |

> **Note:** The multivariate model uses a stratified split (`stratify=y`), while the univariate model uses a standard split, so the two test sets are not identical.

### Confusion Matrix — Univariate Model

|  | Predicted: Not Survived | Predicted: Survived |
|---|---|---|
| **Actual: Not Survived** | 100 | 5 |
| **Actual: Survived** | 57 | 17 |

### Confusion Matrix — Multivariate Model

|  | Predicted: Not Survived | Predicted: Survived |
|---|---|---|
| **Actual: Not Survived** | 95 | 15 |
| **Actual: Survived** | 20 | 49 |

### Multivariate Model Coefficients

| Feature | Coefficient |
|---|---|
| `sex` (encoded) | -2.577 |
| `pclass` | -1.068 |
| `sibsp` | -0.293 |
| `age` | -0.038 |
| `fare` | 0.003 |

---

## 🔍 Key Findings

* The multivariate model is about **15 percentage points more accurate** than the fare-only model.
* The fare-only model predicts "did not survive" for almost every passenger, so it misses most actual survivors (57 of 74).
* **Sex** has the strongest influence on survival, followed by **passenger class**.
* Fare on its own is a weak predictor, because much of its information overlaps with passenger class.

---

## 📁 Project Structure

```text
Titanic-Logistic-Regression/
│
├── titanic_logistic_regression.ipynb
│
└── README.md
```

---

## 🚀 How to Run

### Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Titanic-Logistic-Regression.git
```

### Navigate to the project

```bash
cd Titanic-Logistic-Regression
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Open the notebook

```bash
jupyter notebook
```

Then open:

```text
notebook/titanic_logistic_regression.ipynb
```

---

## 📦 Requirements

Create a `requirements.txt` file containing:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

Install the required libraries with:

```bash
pip install -r requirements.txt
```

---

## 💡 Key Learning Outcomes

Through this project, I learned how to:

* Inspect a dataset and handle missing values
* Encode categorical variables for machine learning
* Build and train Logistic Regression models with Scikit-learn
* Split data into training and testing sets
* Evaluate a classifier using accuracy and a confusion matrix
* Interpret model coefficients to find the most influential features
* Compare a single-feature model against a multi-feature model

---

## 👤 Author

**Satyajit Pradhan**

B.Tech Computer Science Student | Aspiring Data Analyst
