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
    
| Model               | Accuracy  | AUC       | Precision | Recall    | F1 Score  | MCC       |
| ------------------- | --------- | --------- | --------- | --------- | --------- | --------- |
| Logistic Regression | 0.687     | 0.734     | 0.702     | 0.780     | 0.739     | 0.353     |
| Decision Tree       | 0.678     | 0.674     | 0.720     | 0.710     | 0.715     | 0.345     |
| KNN                 | 0.678     | 0.674     | 0.720     | 0.710     | 0.715     | 0.345     |
| Naive Bayes         | 0.616     | 0.658     | 0.695     | 0.578     | 0.631     | 0.241     |
| Random Forest       | 0.736     | 0.793     | 0.731     | 0.846     | 0.785     | 0.456     |
| XGBoost             | 0.738     | 0.813     | 0.772     | 0.766     | 0.769     | 0.467     |

  (Metrics are computed on the test dataset.)

**c.3 Model Observations**

 | **ML Model Name**             | **Observation about Model Performance**                                                                                                                                                        |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression**       | Served as a strong baseline model with good recall, indicating its ability to correctly identify coupon acceptance cases, but showed limited discriminative power compared to ensemble models. |
| **Decision Tree**             | Captured non-linear relationships in the data but showed moderate performance and signs of overfitting, resulting in lower AUC and MCC values.                                                 |
| **K-Nearest Neighbors (KNN)** | Demonstrated similar performance to Decision Tree, indicating sensitivity to feature scaling and limited effectiveness on high-dimensional encoded data.                                       |
| **Naive Bayes (Gaussian)**    | Fast and simple probabilistic model, but strong feature independence assumptions led to the weakest overall performance across most evaluation metrics.                                        |
| **Random Forest (Ensemble)**  | Achieved strong performance with the highest recall, effectively identifying coupon acceptance cases while reducing overfitting through ensemble learning.                                     |
| **XGBoost (Ensemble)**        | Delivered the best overall performance with the highest AUC and MCC, providing the most balanced and robust predictions across all evaluated metrics.                                          |

**d. Streamlit Web Application**

  An interactive Streamlit web application was developed and deployed using Streamlit Community Cloud.
  **Key Features:**
  CSV file download option (test data)
  CSV file upload option (test data)
  Dropdown menu for model selection
  Prediction Threshold Scrollbar
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
|-- data/
|   |--in-vehicle-coupon-recommendation.csv
|   |--test_data_streamlit.csv

**f. Deployment**

The application is deployed using Streamlit Community Cloud and can be accessed through the live application link provided in the submission.

**g. Execution Environment**

  -Models were implemented and executed on BITS Virtual Lab
  -A screenshot of execution has been included in the final submission PDF as proof

**h. Conclusion**

This project demonstrates a complete end-to-end machine learning workflow, including dataset selection, model implementation, performance evaluation, and web-based deployment.

