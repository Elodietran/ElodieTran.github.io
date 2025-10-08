---
title: "Regression Practice Notebook"
layout: post
---

# Regression Practice Notebook

This notebook-style post demonstrates the steps involved in preparing data, training, and evaluating a **logistic regression model** using Python.  
The dataset used here is the Titanic dataset. The workflow covers data cleaning, imputation, model training, evaluation, and feature importance analysis.

---

## In [1]: Import Libraries

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
```

---

## In [2]: Load and Display the Data

```python
data = pd.read_csv('train.csv')
data.head()
```

We start by loading the Titanic dataset and creating a backup copy for safety.

```python
data_original = data.copy()
```

---

## In [3]: Handle Missing Values in Age

We populate missing `Age` values using the average age grouped by `Sex` and `Pclass`.

```python
data['Age'] = data.groupby(['Sex', 'Pclass'], group_keys=False)['Age'].apply(lambda x: x.fillna(x.mean()))
```

Visualize the distributions before and after imputation:

```python
fig, axes = plt.subplots(1, 2, figsize=(6, 8), sharey=True)
sns.histplot(data_original['Age'], ax=axes[0], kde=True, color='red').set_title('Before Imputation')
sns.histplot(data['Age'], kde=True, ax=axes[1], color='green').set_title('After Imputation')
plt.show()
```

---

### Explanation: Why Not Use the "Survived" Field?

Using the `Survived` field to fill in missing `Age` values would cause **data leakage**, meaning the model indirectly learns information from the target variable, making evaluation unrealistic.

---

### Question
> What does the `plt.subplots` function do?  
> How could you find out more about it?

---

## Step 1: Convert Categorical Variables to Numerical

```python
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)
```

**Why?**  
Regression algorithms require numerical input, so categorical variables must be converted into numbers.

---

### Question
> What does the `pd.get_dummies` function do?  
> How would you find out?

---

## Step 2: Define Features and Target Variable

```python
X = data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']]
y = data['Survived']
```

This separates the **input features (X)** from the **target variable (y)**.

---

## Step 3: Split Data into Training and Testing Sets

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Separating the data allows validation on unseen samples to check how well the model generalizes.

---

## Step 4: Train the Logistic Regression Model

```python
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
```

> ⚠️ **Note:** If you see a `ConvergenceWarning`, consider increasing `max_iter` or scaling the data.

---

### Theory: `max_iter` Parameter

This sets the maximum number of iterations for the optimization process. If the model doesn’t converge within this limit, it stops early.

---

### Question
> Why are we using a logistic regression model in this situation?

---

## Step 5: Make Predictions

```python
y_pred = model.predict(X_test)
```

---

## Step 6: Evaluate Model Performance

```python
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:\n', conf_matrix)
```

Example Output:

```
Accuracy: 0.86
Confusion Matrix:
[[46  8]
 [ 5 31]]
```

---

### Explanation: Confusion Matrix and Accuracy Score

- **Confusion Matrix** — shows correct vs incorrect predictions.  
- **Accuracy Score** — percentage of correct predictions.

---

## Step 7: Calculate Feature Importance

```python
feature_importance = model.coef_[0]
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(6,4))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.show()
```

**Interpretation:**  
- Positive importance → increases probability of survival  
- Negative importance → decreases probability of survival

---

## Step 8: Apply Model on Test Data

```python
test_data = pd.read_csv('test.csv')
test_data['Age'] = test_data.groupby(['Sex', 'Pclass'], group_keys=False)['Age'].apply(lambda x: x.fillna(x.mean()))
test_data['Fare'] = test_data.groupby(['Sex', 'Pclass'], group_keys=False)['Fare'].apply(lambda x: x.fillna(x.mean()))

test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'], drop_first=True)
test_data = test_data.reindex(columns=X.columns, fill_value=0)

test_predictions = model.predict(test_data)
test_data['Survived_predicted'] = test_predictions
```

---

### Practical Example

Create a new column called `FamilySize` and build a regression model using:
`Pclass`, `Age`, `FamilySize`, `Fare`, `Sex_male`, `Embarked_Q`, and `Embarked_S`.

```python
data['FamilySize'] = data['SibSp'] + data['Parch']
```

Then, repeat the model training process using the updated feature set.

---

*End of Notebook*
