import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


warnings.filterwarnings("ignore")


file_path = 'bengaluru_rainfall.csv'  
data = pd.read_csv(file_path)


print("Dataset Overview:")
print(data.head())


data.columns = data.columns.str.strip()


print("\nColumns in the dataset:")
print(data.columns)


data = data.dropna()


data['El NiNo (Y/N)'] = data['El NiNo (Y/N)'].map({'Y': 1, 'N': 0})
data['La Nina (Y/N)'] = data['La Nina (Y/N)'].map({'Y': 1, 'N': 0})


threshold = 2000  
data['Flood'] = data['Total'].apply(lambda x: 1 if x > threshold else 0)

X = data[['Year', 'Total', 'El NiNo (Y/N)', 'La Nina (Y/N)']]
y = data['Flood']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


knn = KNeighborsClassifier(n_neighbors=3)


knn.fit(X_train_scaled, y_train)


y_pred = knn.predict(X_test_scaled)


print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))



plt.figure(figsize=(10, 6))
sns.histplot(data['Total'], bins=30, kde=True)
plt.title('Distribution of Annual Rainfall in Bengaluru')
plt.xlabel('Annual Rainfall (mm)')
plt.ylabel('Frequency')
plt.show()



example_year = 2023
example_rainfall = 2200  
example_el_nino = 1  
example_la_nina = 0  


input_data = scaler.transform([[example_year, example_rainfall, example_el_nino, example_la_nina]])


flood_prediction = knn.predict(input_data)

print("\nPrediction for Bengaluru in 2023 with 2200 mm rainfall:")
if flood_prediction[0] == 1:
    print("Warning: Flood predicted.")
else:
    print("No flood predicted.")
