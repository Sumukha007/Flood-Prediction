import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Ignore warnings
warnings.filterwarnings("ignore")

# Load the dataset
file_path = 'bengaluru_rainfall.csv'  # Update the file path accordingly
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
print("Dataset Overview:")
print(data.head())

# Ensure consistency in column names (e.g., remove leading/trailing spaces)
data.columns = data.columns.str.strip()

# Display the columns in the dataset
print("\nColumns in the dataset:")
print(data.columns)

# Handling missing values
data = data.dropna()

# Convert categorical columns to numeric
data['El NiNo (Y/N)'] = data['El NiNo (Y/N)'].map({'Y': 1, 'N': 0})
data['La Nina (Y/N)'] = data['La Nina (Y/N)'].map({'Y': 1, 'N': 0})

# Define flood condition
threshold = 2000  # Example threshold for flood conditions (in mm)
data['Flood'] = data['Total'].apply(lambda x: 1 if x > threshold else 0)

# Features: Year, Total Rainfall, El NiNo, La Nina; Target: Flood
X = data[['Year', 'Total', 'El NiNo (Y/N)', 'La Nina (Y/N)']]
y = data['Flood']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test_scaled)

# Evaluate the model
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot distribution of annual rainfall
plt.figure(figsize=(10, 6))
sns.histplot(data['Total'], bins=30, kde=True)
plt.title('Distribution of Annual Rainfall in Bengaluru')
plt.xlabel('Annual Rainfall (mm)')
plt.ylabel('Frequency')
plt.show()


# Example Prediction: Predict flood for a specific year and rainfall
example_year = 2023
example_rainfall = 2200  # Example rainfall in mm
example_el_nino = 1  # Example El NiNo status
example_la_nina = 0  # Example La Nina status

# Prepare the input data
input_data = scaler.transform([[example_year, example_rainfall, example_el_nino, example_la_nina]])

# Predict the flood condition
flood_prediction = knn.predict(input_data)

print("\nPrediction for Bengaluru in 2023 with 2200 mm rainfall:")
if flood_prediction[0] == 1:
    print("Warning: Flood predicted.")
else:
    print("No flood predicted.")
