import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_csv(r"C:\Users\BIT\OneDrive\Desktop\heat attack predection\archive\heart.csv")
df.columns = df.columns.str.strip()

# Target
X = pd.get_dummies(df.drop("HeartDisease", axis=1), drop_first=True)
y = df["HeartDisease"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train KNN
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Save files ✅
joblib.dump(model, "KNN_heart.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), 'columns.pkl')


print("✅ Model, scaler, and columns saved successfully")
