"""Week 4: Machine Learning Model Development and Evaluation
Educational example using the scikit-learn Breast Cancer Wisconsin dataset.
Install: pip install pandas numpy matplotlib scikit-learn
Run: python analysis.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

OUT="week4_outputs"
os.makedirs(OUT,exist_ok=True)
data=load_breast_cancer(as_frame=True)
X=data.data.copy()
# In this scikit-learn dataset, target 0=malignant, 1=benign.
# Recode malignant as positive class (1) for clearer recall interpretation.
y=(data.target==0).astype(int)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

model=Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler()),
    ("classifier",LogisticRegression(max_iter=3000,random_state=42))
])
model.fit(X_train,y_train)
pred=model.predict(X_test)
prob=model.predict_proba(X_test)[:,1]
print("Accuracy:",accuracy_score(y_test,pred))
print("Precision:",precision_score(y_test,pred))
print("Recall:",recall_score(y_test,pred))
print("F1:",f1_score(y_test,pred))
fpr,tpr,_=roc_curve(y_test,prob)
print("ROC-AUC:",auc(fpr,tpr))
print("Confusion matrix (labels [benign, malignant]):\\n",confusion_matrix(y_test,pred,labels=[0,1]))

# Baseline comparison
base=DummyClassifier(strategy="most_frequent").fit(X_train,y_train)
print("Baseline accuracy:",accuracy_score(y_test,base.predict(X_test)))
# Save model metrics and predictions
pd.DataFrame({"actual_malignant":y_test,"predicted_malignant":pred,"malignant_probability":prob}).to_csv(os.path.join(OUT,"test_predictions.csv"),index=False)
pd.DataFrame([{"accuracy":accuracy_score(y_test,pred),"precision":precision_score(y_test,pred),"recall":recall_score(y_test,pred),"f1":f1_score(y_test,pred),"roc_auc":auc(fpr,tpr)}]).to_csv(os.path.join(OUT,"metrics.csv"),index=False)
