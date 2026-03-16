#  Telco Customer Churn Prediction & Risk Analysis Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](LİNKİNİZ_BURAYA) 
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)

## ・ Project Overview
This project delivers an end-to-end Machine Learning solution designed to predict customer turnover (churn) for a telecommunications company. It features a high-performance **XGBoost** model integrated into a real-time, interactive **Streamlit** dashboard, allowing business stakeholders to perform instant risk assessments.

## ・ Technical Architecture & Methodology

### 1. Data Processing & Feature Engineering
- **Handling Class Imbalance:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to address the dataset's significant churn skewness.
- **Pipeline Integration:** Implemented a robust preprocessing pipeline using **StandardScaler** for numerical features and **One-Hot Encoding** for categorical variables.
- **Complexity Management:** Managed feature mismatching dynamically within the application to ensure model-data alignment during real-time inference.

### 2. Model Performance
I evaluated multiple algorithms (Logistic Regression, Random Forest, XGBoost) and selected **XGBoost** for its superior performance in non-linear classification tasks.
- **Metrics:** Precision, Recall, and F1-Score were optimized to minimize False Negatives (identifying at-risk customers is the priority).
- **Inference:** Model outputs include both a binary classification and a **probability score**, providing a granular view of customer risk.

### 3. Deployment
- **UI/UX:** Built a professional-grade dashboard with **Streamlit**, featuring custom CSS for high-contrast readability and interactive widgets.
- **Real-time Engine:** The dashboard dynamically maps user inputs to the model's high-dimensional feature space (20+ features) for immediate prediction.

## ・ Tech Stack
- **Languages:** Python (Pandas, NumPy, Scikit-Learn)
- **Machine Learning:** XGBoost, Joblib
- **Frontend/Deployment:** Streamlit, CSS3
- **DevOps:** GitHub, Conda Environment Management

## ・ Dashboard Preview
<img width="1284" height="687" alt="image" src="https://github.com/user-attachments/assets/22a68496-333a-4d54-8232-26452ac45730" />
<img width="1269" height="695" alt="image" src="https://github.com/user-attachments/assets/fabceb76-18e1-4be1-bb27-4d103730401e" />


## ・ How to Run Locally
1. Clone the repository:
```bash
   git clone [https://github.com/USERNAME/Churn_App.git](https://github.com/USERNAME/Churn_App.git)
```
2. Install dependencies:
   
```bash
pip install -r requirements.txt
```
3. Launch the dashboard:

```bash
streamlit run app.py
```

   
