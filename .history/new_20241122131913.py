# new.py

# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Step 2: Load the datasets
# Assuming the datasets are in CSV format and located in the same directory
train_data = pd.read_csv('Train.csv')
validation_data = pd.read_csv('Validation.csv')
test_data = pd.read_csv('Test.csv')

# Step 3: Data Preprocessing
# Handle missing values, encode categorical variables, and normalize features if necessary
def preprocess_data(data):
    # Step 3.1: Handle Missing Values
    # Fill missing numerical values with the median
    for column in data.select_dtypes(include=['float64', 'int64']).columns:
        data[column].fillna(data[column].median(), inplace=True)
    
    # Fill missing categorical values with the mode
    for column in data.select_dtypes(include=['object']).columns:
        data[column].fillna(data[column].mode()[0], inplace=True)
    
    # Step 3.2: Encode Categorical Variables
    # Convert categorical variables to dummy variables
    data = pd.get_dummies(data, drop_first=True)
    
    # Step 3.3: Normalize Features
    # Normalize numerical features using StandardScaler
    scaler = StandardScaler()
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    
    return data



train_data = preprocess_data(train_data)
validation_data = preprocess_data(validation_data)
test_data = preprocess_data(test_data)

# Step 4: Split features and target labels
# drop ID and Class(Target)
X_train = train_data.drop(['ID', 'Class(Target)'], axis=1)
y_train = train_data['Class(Target)']
X_val = validation_data.drop('Class(Target)', axis=1)
y_val = validation_data['Class(Target)']
X_test = test_data  # Test data has no target column

# Step 5: Model Selection and Training
# import all models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier  
# use different models and compare their performance
models = [RandomForestClassifier(), GradientBoostingClassifier(), SVC(), KNeighborsClassifier()]

# Step 6: Hyperparameter Tuning
# Example: Using GridSearchCV for hyperparameter optimization
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Step 7: Evaluate the Model
# Evaluate on validation set
best_model = grid_search.best_estimator_
y_val_pred = best_model.predict(X_val)
print("Validation Set Evaluation:")
print(classification_report(y_val, y_val_pred))
print(confusion_matrix(y_val, y_val_pred))

# Step 8: Impute Missing Labels in Test Set
y_test_pred = best_model.predict(X_test)
test_data['predicted_target'] = y_test_pred

# Step 9: Save the Imputed Test Dataset
test_data.to_csv('Test_Imputed.csv', index=False)

# Step 10: Prepare the Assignment Report
# This step involves writing a detailed report as per the guidelines in readme.md
# Include model details, evaluation results, and key observations

# Note: The above code is a template and may require adjustments based on the actual dataset and requirements. 