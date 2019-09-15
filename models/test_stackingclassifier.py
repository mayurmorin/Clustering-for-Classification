import numpy as np
from StackingClassifier import StackingClassifier
from sklearn import linear_model, metrics
from sklearn import tree
import random
import pandas as pd
#Create an artificial data for demonstration
#reading Data
X = pd.read_csv('../data/raw/Data Cleaning.csv')
y = pd.read_excel('../data/raw/Training outputs.xlsx')

X.drop(columns=['Unnamed: 0'],axis=1,inplace=True)
y.drop(columns=['Unnamed: 0'],axis=1,inplace=True)

#use ONEGO technique to create stacking model
stacking_classifier = StackingClassifier(
    base_classifiers=[
        linear_model.SGDClassifier(loss='log', random_state=0),
        linear_model.LogisticRegression(random_state=0),
        tree.DecisionTreeClassifier(random_state=0)
    ],
    combiner=linear_model.LogisticRegression(),
    technique=StackingClassifier.ONEGO
)

stacking_classifier.fit(X, y)
predicted_y_proba = stacking_classifier.predict_proba(X)
print(metrics.roc_auc_score(y, predicted_y_proba))

#use OUTOFFOLDS technique to create stacking model
stacking_classifier = StackingClassifier(
    base_classifiers=[
        linear_model.SGDClassifier(loss='log', random_state=0),
        linear_model.LogisticRegression(random_state=0),
        tree.DecisionTreeClassifier(random_state=0)
    ],
    combiner=linear_model.LogisticRegression(),
    technique=StackingClassifier.OUTOFFOLDS
)

stacking_classifier.fit(X, y)
predicted_y = stacking_classifier.predict(X)
predicted_y_proba = stacking_classifier.predict_proba(X)
#print("Accuracy:", metrics.roc_auc_score(y, predicted_y_proba))

