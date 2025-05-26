# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import Necessary Libraries and Load Data
2.Split Dataset into Training and Testing Sets
3.Train the Model Using Stochastic Gradient Descent (SGD)
4.Make Predictions and Evaluate Accuracy
5.Generate Confusion Matrix
```
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Pranav Krishna T
RegisterNumber:  212224040241
*/

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv('/content/drive/MyDrive/datasetml/spam.csv', encoding='latin-1')
data = data[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})

print("data.head()\n")
print(data.head())
print("\ndata.info()\n")
print(data.info())
print("\ndata.isnull().sum()\n")
print(data.isnull().sum())
print()

data['label'] = data['label'].map({'ham': 0, 'spam': 1})

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(data['text'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

print("y_pred")
print(y_pred)
print("\naccuracy")
print(accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

## Output:

![image](https://github.com/user-attachments/assets/3345a7ad-a748-443b-927e-722c5082fe27)


![image](https://github.com/user-attachments/assets/2acfe025-cd09-40cc-aac7-6bb23ab5921e)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
