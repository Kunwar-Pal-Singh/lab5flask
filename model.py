import pandas as pd

# Load the dataset
file_path = 'Fish.csv'
df = pd.read_csv(file_path)

 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Preprocess the data
X = df.drop(columns=['Species'])
y = df['Species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model
import pickle
model_filename = 'fish_species_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
