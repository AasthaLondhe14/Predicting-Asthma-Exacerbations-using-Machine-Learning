# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("/content/MLProject (2).csv")

# Preprocess the data
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
data['Smoking_Status'] = le.fit_transform(data['Smoking_Status'])
data['Intensity of cough'] = le.fit_transform(data['Intensity of cough'])
data['Asthma_Diagnosis'] = le.fit_transform(data['Asthma_Diagnosis'])

# Define features and target variable
X = data[['Age', 'Gender', 'Smoking_Status', 'Medvalue', 'Intensity of cough']]
y = data['Asthma_Diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the KNN model
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the model on the training data
knn.fit(X_train, y_train)

# Make predictions
predictions = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy of the KNN model:", accuracy)

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Calculate precision
precision = precision_score(y_test, predictions)

# Calculate recall
recall = recall_score(y_test, predictions)

# Calculate F1-score
f1 = f1_score(y_test, predictions)

# Calculate sensitivity
sensitivity = recall

# Calculate specificity
tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
specificity = tn / (tn + fp)

# Display precision, recall, F1-score, sensitivity, specificity, and confusion matrix
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Function to predict asthma diagnosis based on user input
def predict_asthma(age, gender, smoking_status, medvalue, intensity_cough):
    user_input = pd.DataFrame([[age, gender, smoking_status, medvalue, intensity_cough]], columns=['Age', 'Gender', 'Smoking_Status', 'Medvalue', 'Intensity of cough'])
    prediction = knn.predict(user_input)
    return prediction

# Calculate mean absolute error (MAE)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error (MAE):", mae)

# Calculate mean squared error (MSE)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error (MSE):", mse)

# Calculate R-squared error
from sklearn.metrics import r2_score
r2 = r2_score(y_test, predictions)
print("R-squared Error:", r2)

# User input
age = int(input("Enter your age: "))
gender = int(input("Enter your gender (1 for Male, 0 for Female): "))
smoking_status = int(input("Enter your smoking status (1 for Current Smoker, 0 for Ex-Smoker, -1 for Non-Smoker): "))
medvalue = float(input("Enter your Medvalue: "))
intensity_cough = int(input("Enter your intensity of cough (0 for low, 1 for medium, 2 for high): "))

# Predict asthma diagnosis
prediction = predict_asthma(age, gender, smoking_status, medvalue, intensity_cough)

# Print the prediction
if prediction == 1:
    print("You have asthma.")
else:
    print("You do not have asthma.")
