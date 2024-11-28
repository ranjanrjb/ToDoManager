"""
Author: Hemen Babis
Date: February 2, 2023

Description:
This script is responsible for training the AI model used to predict task priorities. 
The model is trained on example data using a Decision Tree Regressor. It predicts task 
priority based on features such as task importance, days left until due date, and description length.
The trained model is saved as 'task_priority_model.pkl' for later use in the Flask app.
"""
import numpy as np
import joblib
from sklearn.tree import DecisionTreeRegressor

# Example data for training the model (importance, days_left, description_length)
X_train = np.array([[5, 2, 20], [3, 10, 15], [1, 5, 25], [4, 1, 30]])
y_train = np.array([9, 6, 4, 8])  # Priority scores based on user feedback

# Train the model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'task_priority_model.pkl')

print("Model trained and saved as 'task_priority_model.pkl'")
