# Waste Cost Prediction Model 🏭💰

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange)
![Pandas](https://img.shields.io/badge/Pandas-Latest-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📊 Project Overview

This project implements a machine learning model to predict waste costs in manufacturing processes. By analyzing various production factors, the model aims to help industries optimize their processes and reduce waste-related expenses.

## 🚀 Features

- Data preprocessing and cleaning
- Feature engineering and selection
- Implementation of stacking regressor model
- Model evaluation and cross-validation
- Visualization of results

## 🛠️ Installation

git clone https://github.com/dimplefrancis/waste-cost-prediction.git
cd waste-cost-prediction
pip install -r requirements.txt

## 🏃‍♂️ Usage
python main.py

## 📁 Project Structure

waste-cost-prediction/
│
├── data/
│   ├── data_batch.csv
│   └── data_fail.csv
│
├── data_preparation.py
├── feature_engineering.py
├── model.py
├── evaluation.py
├── utils.py
├── main.py
└── README.md

## 📈 Results
![alt text](image.png)
![alt text](image-1.png)
Interpretation:
•	Mean Squared Error (MSE): The MSE value is significantly lower after outlier removal. This indicates that the model's predictions are much closer to the actual values, meaning the average squared difference between the predicted and actual values has reduced substantially.
•	R-squared (R2): The R2 score is approximately 0.966, indicating that the model explains about 96.58% of the variance in the target variable after removing outliers. This is a substantial improvement compared to the initial results, showing that the model now captures most of the variation in the data.
Implications:
•	Performance Improvement: The significant reduction in MSE and the increase in R2 score demonstrate that the model's performance has greatly improved after handling outliers. This suggests that outliers were previously distorting the model's predictions.
•	Model Robustness: The high R2 score indicates a strong fit, suggesting that the model is robust and can effectively predict the target variable for most data points.
After performing cross-validation, the results are as follows:
Cross-Validation R2 Scores:
Fold 1: 0.9616
Fold 2: 0.9563
Fold 3: 0.9418
Fold 4: 0.9567
Fold 5: 0.9488
Mean Cross-Validation R2 Score: 0.9530
Interpretation:
•	Consistent High Performance: The R2 scores from cross-validation are consistently high across all folds, indicating that the model maintains strong performance across different subsets of the data. This consistency is crucial for ensuring that the model generalizes well to unseen data.
•	Mean R2 Score: The mean R2 score of 0.9530 across all folds suggests that, on average, the model explains 95.30% of the variance in the target variable during cross-validation. This further reinforces the model’s robustness and reliability.
Implications:
•	Generalization: The consistent and high cross-validation R2 scores imply that the model generalizes well across different data subsets, making it reliable for real-world applications.
•	Validation of Improvements: The improved cross-validation scores validate the effectiveness of the steps taken (such as outlier removal and feature selection) to enhance the model.
![alt text](image-2.png)
![alt text](image-3.png)

📝 License
This project is MIT licensed.