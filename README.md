# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.
2.Upload and read the dataset.
3.Check for any null values using the isnull() function.
4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5.Find the accuracy of the model and predict the required values by importing the required module from sklearn. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: AFSAR JUMAIL S
RegisterNumber:  212222240004
*/
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()
x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y = data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
### Data Head:
![image](https://github.com/Afsarjumail/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118343395/abfdde85-1bc2-434e-b574-d140ba49d4bf)


### Information:
![image](https://github.com/Afsarjumail/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118343395/1c9f6647-e50f-461e-8719-1785d82d234e)

### Null dataset:
![image](https://github.com/Afsarjumail/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118343395/c59b9c68-1e09-452d-8812-cd1ccdf78529)

### Value_counts():
![image](https://github.com/Afsarjumail/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118343395/7ce80162-ab63-4f1f-9976-1b7756dca2e7)


### Data Head:
![image](https://github.com/Afsarjumail/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118343395/73854566-0de6-4206-a5a4-6d09ab7d662f)


### x.head():
![image](https://github.com/Afsarjumail/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118343395/849c0a38-a1d0-4b56-b707-7ebf21688be6)


### Accuracy:
![image](https://github.com/Afsarjumail/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118343395/2d672eda-13c8-4cbd-8aad-081874152fc5)


### Data Prediction:
![image](https://github.com/Afsarjumail/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118343395/ac0d0c50-67e4-4c5d-85bd-3c0495920860)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
