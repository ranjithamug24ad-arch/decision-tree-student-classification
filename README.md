# Decision Tree Student Classification

## Machine Learning Lab – Experiment 3

This repository contains the implementation of **Experiment 3 – Decision Tree Classification** for the Machine Learning Lab.

The objective of this experiment is to build a machine learning classification model using the **Decision Tree Classifier** algorithm. A custom student dataset is used instead of the commonly used Iris dataset. The model predicts whether a student is likely to **Pass** or **Fail** based on selected student-related features.

---

## 📌 Experiment Details

**Experiment Number:** 3
**Experiment Name:** Decision Tree Classification
**Domain:** Machine Learning
**Programming Language:** Python
**Algorithm:** Decision Tree Classifier
**Task:** Binary Classification

---

## 🎯 Aim

To implement a **Decision Tree Classification** model using Python and evaluate its performance using a **Confusion Matrix, Accuracy Score, and Classification Report**.

---

## 📖 Introduction

Decision Tree is a supervised machine learning algorithm that can be used for both classification and regression problems.

A Decision Tree makes predictions by repeatedly splitting the dataset based on feature values. The resulting structure looks similar to a tree, where:

* The **root node** represents the first decision.
* The **internal nodes** represent conditions or tests on features.
* The **branches** represent the outcomes of those conditions.
* The **leaf nodes** represent the final predicted class.

In this experiment, the Decision Tree algorithm is used as a classification model to predict a student's academic result.

The model learns patterns from the training data and uses those patterns to classify unseen test data into one of two categories:

* **Pass**
* **Fail**

---

## 🧠 Problem Statement

Student performance can be influenced by several factors, including study time and attendance.

In this experiment, a machine learning model is developed to predict the student's result using:

1. Age
2. Study Hours
3. Attendance Percentage

The target variable is the student's final result:

* `Pass`
* `Fail`

The dataset is divided into training and testing sets. The Decision Tree model is trained using the training data and evaluated using the test data.

---

## 📊 Dataset Description

A custom dataset named `decision_tree_student_dataset.csv` is used for this experiment.

The dataset contains the following columns:

| Column     | Description                    | Type        |
| ---------- | ------------------------------ | ----------- |
| Age        | Age of the student             | Integer     |
| StudyHours | Number of hours spent studying | Integer     |
| Attendance | Attendance percentage          | Integer     |
| Result     | Student's final result         | Categorical |

### Features

The input features used by the model are:

```text
Age
StudyHours
Attendance
```

### Target

The target variable is:

```text
Result
```

The target contains two classes:

```text
Pass
Fail
```

---

## 📁 Project Structure

```text
decision-tree-student-classification/
│
├── decision_tree_student_dataset.csv
│
├── decision_tree.py
│
└── README.md
```

### File Description

#### `decision_tree_student_dataset.csv`

Contains the custom student dataset used for training and testing the machine learning model.

#### `decision_tree.py`

Contains the Python implementation of the Decision Tree Classification algorithm.

#### `README.md`

Contains the documentation and explanation of the experiment.

---

## 🛠️ Technologies Used

The following technologies and Python libraries are used:

* **Python**
* **Pandas**
* **Scikit-learn**

### Python

Python is used to implement the machine learning algorithm and perform data processing.

### Pandas

Pandas is used to:

* Load the CSV dataset
* Store the dataset in a DataFrame
* Select features and target values
* Display the dataset

### Scikit-learn

Scikit-learn is used for:

* Splitting the dataset
* Creating the Decision Tree model
* Training the model
* Making predictions
* Calculating accuracy
* Generating the confusion matrix
* Generating the classification report

---

## ⚙️ Installation

Make sure Python is installed on your computer.

You can check the Python version using:

```bash
python --version
```

Install the required Python libraries using:

```bash
pip install pandas scikit-learn
```

---

## 🚀 How to Run the Project

### Step 1 – Clone the Repository

```bash
git clone https://github.com/your-username/decision-tree-student-classification.git
```

### Step 2 – Open the Project Folder

```bash
cd decision-tree-student-classification
```

### Step 3 – Install Dependencies

```bash
pip install pandas scikit-learn
```

### Step 4 – Run the Python Program

```bash
python decision_tree.py
```

The program will load the CSV dataset, train the Decision Tree model, make predictions, and display the evaluation results.

---

## 🔬 Methodology

The experiment follows the steps below.

### Step 1 – Import Libraries

The required Python libraries are imported.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
```

---

### Step 2 – Load the Dataset

The CSV file is loaded using Pandas.

```python
df = pd.read_csv("decision_tree_student_dataset.csv")
```

The dataset is then displayed using:

```python
print(df)
```

---

### Step 3 – Separate Features and Target

The input features are selected:

```python
X = df[["Age", "StudyHours", "Attendance"]]
```

The target variable is selected:

```python
y = df["Result"]
```

Here:

* `X` contains the independent variables.
* `y` contains the dependent variable.

---

### Step 4 – Split the Dataset

The dataset is divided into training and testing data.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

In this experiment:

* 80% of the data is used for training.
* 20% of the data is used for testing.
* `random_state=42` ensures reproducible results.

---

### Step 5 – Create the Decision Tree Model

A Decision Tree Classifier is created:

```python
dt_model = DecisionTreeClassifier(random_state=42)
```

---

### Step 6 – Train the Model

The model is trained using the training dataset.

```python
dt_model.fit(X_train, y_train)
```

During training, the Decision Tree identifies patterns in the input features and their corresponding results.

---

### Step 7 – Make Predictions

The trained model is used to predict the results of the test dataset.

```python
y_pred = dt_model.predict(X_test)
```

The predicted values are then compared with the actual values.

---

## 📈 Model Evaluation

The performance of the model is evaluated using the following metrics.

### 1. Confusion Matrix

A confusion matrix shows how many predictions were correctly or incorrectly classified.

```python
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
```

For binary classification, the confusion matrix contains values representing:

* True Positive
* True Negative
* False Positive
* False Negative

It helps us understand the types of errors made by the classifier.

---

### 2. Accuracy Score

Accuracy represents the percentage of predictions that were correctly classified.

```python
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
```

The general formula for accuracy is:

```text
Accuracy = Correct Predictions / Total Predictions
```

A higher accuracy indicates that the model correctly classified a larger proportion of the test samples.

---

### 3. Classification Report

A classification report provides additional performance measures.

```python
print(classification_report(y_test, y_pred))
```

The report includes:

* Precision
* Recall
* F1-score
* Support

#### Precision

Precision indicates how many of the samples predicted as a particular class actually belong to that class.

#### Recall

Recall indicates how many actual samples of a class were correctly identified.

#### F1-score

F1-score combines precision and recall into a single metric.

#### Support

Support represents the number of actual samples belonging to each class.

---

## 💻 Complete Python Program

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load dataset
df = pd.read_csv("decision_tree_student_dataset.csv")

# Display dataset
print("Dataset:")
print(df)

# Separate features and target
X = df[["Age", "StudyHours", "Attendance"]]
y = df["Result"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)

# Train the model
dt_model.fit(X_train, y_train)

# Make predictions
y_pred = dt_model.predict(X_test)

# Display actual and predicted values
print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(y_pred)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

---

## 🧪 Expected Output

After executing the program, the output will contain:

```text
Dataset:
    Age  StudyHours  Attendance Result
0    22           2          60   Fail
1    25           3          65   Fail
2    28           4          70   Fail
...

Actual Values:
[...]

Predicted Values:
[...]

Confusion Matrix:
[[... ...]
 [... ...]]

Accuracy:
...

Classification Report:
              precision    recall  f1-score   support

        Fail       ...       ...       ...       ...
        Pass       ...       ...       ...       ...

    accuracy                           ...       ...
   macro avg       ...       ...       ...       ...
weighted avg       ...       ...       ...       ...
```

The exact prediction values and evaluation metrics may depend on the dataset split and model configuration.

---

## 🌳 Why Decision Tree?

Decision Trees are useful because they are:

* Easy to understand
* Easy to implement
* Suitable for classification tasks
* Capable of handling numerical features
* Relatively easy to interpret

For this experiment, a Decision Tree is suitable because the goal is to classify students into two result categories.

---

## 🔍 Advantages

Some advantages of Decision Tree Classification are:

1. **Simple to understand**
   The decision-making process can be represented as a tree.

2. **Easy to implement**
   Scikit-learn provides a simple implementation.

3. **Works with numerical data**
   Features such as age, study hours, and attendance can be used directly.

4. **No feature scaling required**
   Decision Trees generally do not require normalization or standardization of input features.

5. **Interpretable results**
   The model can provide a visual representation of its decision-making process.

---

## ⚠️ Limitations

Decision Trees also have some limitations:

1. They can overfit the training data.
2. A very deep tree can become unnecessarily complex.
3. Small changes in training data can sometimes produce a different tree.
4. A poorly configured tree may not generalize well to unseen data.

For larger datasets, techniques such as pruning, Random Forest, or Gradient Boosting can be considered.

---

## 📚 Learning Outcomes

After completing this experiment, the following concepts can be understood:

* Supervised machine learning
* Classification problems
* Decision Tree algorithm
* Training and testing datasets
* Feature and target selection
* Model training
* Prediction
* Confusion Matrix
* Accuracy Score
* Precision
* Recall
* F1-score
* Classification Report

---

## 📝 Experiment Conclusion

In this experiment, a **Decision Tree Classification model** was implemented using Python and Scikit-learn.

A custom student dataset was used to predict whether a student would **Pass or Fail** based on age, study hours, and attendance.

The dataset was divided into training and testing sets. The Decision Tree model was trained using the training data and used to make predictions on the test data.

The performance of the model was evaluated using a **Confusion Matrix, Accuracy Score, and Classification Report**.

Thus, the Decision Tree Classification algorithm was successfully implemented and evaluated for the student result prediction problem.

---

## 👨‍💻 Author

Ranjitha M



---

## 📄 License

This project is created for **educational and academic purposes** as part of a Machine Learning Lab experiment.
