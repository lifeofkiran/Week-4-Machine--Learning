WEEK 4: MACHINE LEARNING MODEL DEVELOPMENT AND EVALUATION

Objective:
Build and evaluate a basic machine learning classifier using a public dataset.

Dataset:
Breast Cancer Wisconsin (Diagnostic) dataset bundled with scikit-learn. Target is recoded so malignant=1 (positive class), benign=0. This is only for educational model-evaluation practice and must not be used for medical decisions.

Pipeline:
1. Split data into training and test sets (80/20, stratified, random_state=42).
2. Impute any missing numeric values using the training-set median.
3. Standardize features.
4. Train logistic regression.
5. Evaluate using accuracy, precision, recall, F1, confusion matrix and ROC-AUC.
6. Compare with a majority-class baseline.

Install:
pip install pandas numpy matplotlib scikit-learn
Run:
python analysis.py

Important:
This project is educational only and is not a medical diagnostic tool. Evaluation metrics depend on the dataset and split. Recall for the malignant class is included because false negatives are important in this example.
