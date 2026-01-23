**In-Vehicle Coupon Recommendation – Machine Learning Classification**

**a. Problem Statement**

The goal of this project is to build and evaluate multiple machine learning classification models to predict whether a customer will accept or reject a coupon while traveling in a vehicle.
Accurate prediction of coupon acceptance can help businesses deliver personalized and timely promotional offers, thereby improving customer engagement and conversion rates.
This project implements classical and ensemble classification algorithms and deploys them using an interactive Streamlit web application.

**b. Dataset Description**
- Dataset Name: In-Vehicle Coupon Recommendation Dataset
- Source: UCI Machine Learning Repository
- Problem Type: Binary Classification
- Target Variable: Y (1 – Coupon Accepted, 0 – Coupon Rejected)

**Dataset Characteristics:**
  - Number of Instances: ~12,000
  - Number of Features: 20+
  - **Feature Types:**
    - Categorical (e.g., destination, weather, time, coupon type)
    - Numerical (e.g., age, income, temperature)

The dataset contains contextual information about the user, vehicle, environment, and coupon attributes, making it suitable for evaluating multiple machine learning models.

**c. Machine Learning Models Used and Evaluation**

All models were trained and tested on the same dataset and train–test split to ensure fair comparison.
Each model was evaluated using six standard classification metrics.

**c.1 Models Implemented**

    1.Logistic Regression
  
    2.Decision Tree Classifier
  
    3.K-Nearest Neighbors (KNN)
  
    4.Naive Bayes (Gaussian)
  
    5.Random Forest Classifier (Ensemble)
  
    6.XGBoost Classifier (Ensemble)
  
**c.2 Performance Comparison Table**
    
  | ML Model                 | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
  | ------------------------ | -------- | --- | --------- | ------ | -------- | --- |
  | Logistic Regression      |          |     |           |        |          |     |
  | Decision Tree            |          |     |           |        |          |     |
  | K-Nearest Neighbors      |          |     |           |        |          |     |
  | Naive Bayes              |          |     |           |        |          |     |
  | Random Forest (Ensemble) |          |     |           |        |          |     |
  | XGBoost (Ensemble)       |          |     |           |        |          |     |
  (Metrics are computed on the test dataset.)

**c.3 Model Observations**

  | ML Model Name            | Observation about Model Performance                                                                                          |
  | ------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
  | Logistic Regression      | Performed reasonably well as a baseline model but struggled to capture complex non-linear relationships present in the data. |
  | Decision Tree            | Captured non-linear patterns but showed signs of overfitting compared to ensemble methods.                                   |
  | K-Nearest Neighbors      | Performance was sensitive to the choice of K and feature scaling, with moderate accuracy overall.                            |
  | Naive Bayes              | Fast and efficient but based on strong independence assumptions, leading to comparatively lower performance.                 |
  | Random Forest (Ensemble) | Achieved strong overall performance by reducing overfitting and handling feature interactions effectively.                   |
  | XGBoost (Ensemble)       | Delivered the best performance across most metrics due to gradient boosting and regularization capabilities.                 |

**d. Streamlit Web Application**

  An interactive Streamlit web application was developed and deployed using Streamlit Community Cloud.
  **Key Features:**
  CSV file upload option (test data)
  Dropdown menu for model selection
  Display of evaluation metrics
  Confusion matrix / classification report for selected model
  The application allows users to experiment with different models and visually analyze their performance.

**e. Repository Structure**

project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- model/
│   │-- logistic_regression.py
│   │-- decision_tree.py
│   │-- knn.py
│   │-- naive_bayes.py
│   │-- random_forest.py
│   │-- xgboost.py
|-- in-vehicle-coupon-recommendation.csv
|-- test_data_streamlit.csv

**f. Deployment**

The application is deployed using Streamlit Community Cloud and can be accessed through the live application link provided in the submission.

**g. Execution Environment**

  -Models were implemented and executed on BITS Virtual Lab
  -A screenshot of execution has been included in the final submission PDF as proof

**h. Conclusion**

This project demonstrates a complete end-to-end machine learning workflow, including dataset selection, model implementation, performance evaluation, and web-based deployment.

