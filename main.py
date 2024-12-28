import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset and assign column names
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
                'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
data = pd.read_csv(url, header=None, names=column_names)

# Data preprocessing
data.replace('?', np.nan, inplace=True)  # Replace missing values marked as '?'
data.dropna(inplace=True)                # Remove rows with missing values
data['ca'] = data['ca'].astype(float)    # Convert 'ca' column to float
data['thal'] = data['thal'].astype(float)  # Convert 'thal' column to float

# Define features (X) and target (y)
X = data.drop('target', axis=1)
y = data['target']

# Convert target variable into binary form (1 for disease, 0 for no disease)
y = y.apply(lambda x: 1 if x > 0 else 0)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define machine learning models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear Regression": LinearRegression(),
    "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
    "Decision Tree Classifier": DecisionTreeClassifier(),
    "Random Forest Classifier": RandomForestClassifier()
}

# Train and evaluate models
for model_name, model in models.items():
    model.fit(X_train, y_train)
    
    # Predict and handle special case for Linear Regression
    if model_name == "Linear Regression":
        predictions = np.round(model.predict(X_test)).astype(int)  # Round predictions to nearest int
    else:
        predictions = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions)
    
    # Display results
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(report)
    print("\n" + "-" * 50 + "\n")

# Test model predictions on 5 sample data points
sample_data = X_test[:5]
for model_name, model in models.items():
    if model_name == "Linear Regression":
        sample_predictions = np.round(model.predict(sample_data)).astype(int)
    else:
        sample_predictions = model.predict(sample_data)
    
    print(f"Sample Predictions for {model_name}: {sample_predictions}")
