# Performance Analysis of Classification Algorithms for Heart Disease Prediction

This project evaluates and compares seven supervised machine learning algorithms to identify the most effective model for predicting cardiovascular disease (CVD) using a heart disease dataset.

## Project Overview

- **Objective:** To determine the best-performing classification algorithm for heart disease prediction.
- **Dataset:** Cardiovascular disease data sourced from [Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset).
- **Algorithms Compared:**  
  - K-Nearest Neighbors  
  - Naive Bayes  
  - Support Vector Machines  
  - Neural Networks (Multilayer Perceptron)  
  - Random Forest Classifier  
  - Decision Tree Classifier  
  - Gradient Boosting Classifier

Among these, the Gradient Boosting Classifier demonstrated superior performance based on metrics such as accuracy, precision, and sensitivity.

## Abstract

Machine learning has become a powerful tool in healthcare, enabling early disease prediction and improved patient outcomes. Early detection of heart disease can lead to better management and treatment, potentially reducing the rising number of deaths caused by heart attacks. This project applies various supervised machine learning techniques to predict heart disease, highlighting the effectiveness of Gradient Boosting in this context.

**Keywords:** Heart disease, Machine learning, Supervised learning, K-Nearest Neighbors, Naive Bayes, Support Vector Machines, Neural Networks, Random Forest, Decision Tree, Gradient Boosting.

## Getting Started

### Prerequisites

- Python version `~=3.11`
- Required packages listed in [PACKAGES.md](PACKAGES.md)

### Installation

1. Clone the repository using Git or GitHub Desktop.
2. Open the project in your preferred IDE.
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Streamlit App

1. Launch the app with:
   ```bash
   streamlit run app/navigation.py
   ```
2. The app will open in your default browser, or you can access it at `localhost:8501`.
3. To perform analysis, upload the dataset downloaded from the link above.

## Additional Resources

- Learn more about [Supervised Learning](https://developers.google.com/machine-learning/intro-to-ml/what-is-ml#supervised_learning).

---

**Created and maintained by [Devendranath Bhavanasi](https://linkedin.com/in/Devendranath-Bhavanasi)**

---
