# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: kavya p
RegisterNumber: 212225240110

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("spam.csv", encoding="latin-1")

df = df[['v1','v2']]
df.columns = ['label','message']

df['label'] = df['label'].map({'ham':0, 'spam':1})
X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english')

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train_vec, y_train)
y_pred = svm_model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

email = ["Congratulations! You have won a free lottery ticket"]

email_vec = vectorizer.transform(email)
prediction = svm_model.predict(email_vec)

if prediction[0] == 1:
    print("Prediction: Spam Mail")
else:
    print("Prediction: Not Spam")
```
# Output:
<img width="622" height="264" alt="image" src="https://github.com/user-attachments/assets/aa3ed496-0e76-4349-b268-50e1845d41f8" />
Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
