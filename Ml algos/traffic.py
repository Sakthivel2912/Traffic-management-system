import torch

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def count_vehicles(image):
    results = model(image)
    return len(results.pandas().xyxy[0])

# Example usage with an image
image = 'path_to_image.jpg'
vehicle_count = count_vehicles(image)
print(f'Vehicle count: {vehicle_count}')

import pandas as pd
import numpy as np

# Simulated data collection
junctions = ['N1'] * 4 + ['N2'] * 3
cameras = ['Cam1', 'Cam2', 'Cam3', 'Cam4', 'Cam5', 'Cam6', 'Cam7']
data = pd.DataFrame({
    'junction': junctions * 50,
    'camera': cameras * 50,
    'vehicle_count': np.random.randint(0, 50, 350),
    'time_of_day': np.random.randint(0, 24, 350),
    'day_of_week': np.random.randint(0, 7, 350),
    'weather_condition': np.random.choice(['Clear', 'Rain', 'Fog'], 350),
    'signal': np.random.choice(['RED', 'YELLOW', 'GREEN_STRAIGHT', 'GREEN_RIGHT', 'GREEN_LEFT'], 350)
})

# Convert categorical data
data['junction'] = data['junction'].map({'N1': 0, 'N2': 1})
data['camera'] = data['camera'].map({
    'Cam1': 0,
    'Cam2': 1,
    'Cam3': 2,
    'Cam4': 3,
    'Cam5': 4,
    'Cam6': 5,
    'Cam7': 6
})
data['weather_condition'] = data['weather_condition'].map({'Clear': 0, 'Rain': 1, 'Fog': 2})
data['vehicle_time_interaction'] = data['vehicle_count'] * data['time_of_day']

# Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = data[['junction', 'camera', 'vehicle_count', 'time_of_day', 'day_of_week', 'weather_condition', 'vehicle_time_interaction']]
X_scaled = scaler.fit_transform(X)
y = data['signal']

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Evaluate Random Forest model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Tuned Model Accuracy: {accuracy:.2f}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification report (Precision, Recall, F1-Score)
report = classification_report(y_test, y_pred, target_names=['RED', 'YELLOW', 'GREEN_STRAIGHT', 'GREEN_RIGHT', 'GREEN_LEFT'])
print("Classification Report:")
print(report)

# XGBoost model
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
print(f'XGBoost Model Accuracy: {xgb_accuracy:.2f}')

# Ensemble model (RandomForest + XGBoost)
voting_clf = VotingClassifier(estimators=[('rf', best_model), ('xgb', xgb)], voting='soft')
voting_clf.fit(X_train, y_train)
ensemble_pred = voting_clf.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
print(f'Ensemble Model Accuracy: {ensemble_accuracy:.2f}')

# Cross-validation to assess model robustness
cv_scores = cross_val_score(voting_clf, X_scaled, y, cv=5, scoring='accuracy')
print(f'Cross-validated Accuracy: {np.mean(cv_scores):.2f}')

# Feature importance (Random Forest)
import matplotlib.pyplot as plt
importances = best_model.feature_importances_
plt.barh(X.columns, importances)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()

# Save the ensemble model
import joblib
joblib.dump(voting_clf, 'traffic_model_ensemble.joblib')

# Flask API for real-time predictions
from flask import Flask, request, jsonify
app = Flask(__name__)
model = joblib.load('traffic_model_ensemble.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    junction_id = data.get('junction_id')
    camera_id = data.get('camera_id')
    vehicle_count = data.get('vehicle_count')
    time_of_day = data.get('time_of_day')
    day_of_week = data.get('day_of_week')
    weather_condition = data.get('weather_condition')
    vehicle_time_interaction = vehicle_count * time_of_day
    
    prediction = model.predict([[junction_id, camera_id, vehicle_count, time_of_day, day_of_week, weather_condition, vehicle_time_interaction]])
    signal = ['RED', 'YELLOW', 'GREEN_STRAIGHT', 'GREEN_RIGHT', 'GREEN_LEFT'][prediction[0]]
    
    return jsonify(signal)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=38888)
