# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree model with pruning parameters
dt = DecisionTreeClassifier(max_depth=5, min_samples_split=5)

# Fit the model on the training data
dt.fit(X_train, y_train)

# Make predictions
predictions = dt.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy of the Decision Tree model with pruning and min_samples_split=5:", accuracy)

# Calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Calculate F1-score
from sklearn.metrics import f1_score
f1 = f1_score(y_test, predictions)
print("F1-score:", f1)

# Calculate precision
from sklearn.metrics import precision_score
precision = precision_score(y_test, predictions)
print("Precision:", precision)

# Calculate recall
from sklearn.metrics import recall_score
recall = recall_score(y_test, predictions)
print("Recall:", recall)

# Calculate sensitivity
sensitivity = recall
print("Sensitivity:", sensitivity)

# Calculate specificity
from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
specificity = tn / (tn + fp)
print("Specificity:", specificity)

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

# Confusion Matrix
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_mat)
