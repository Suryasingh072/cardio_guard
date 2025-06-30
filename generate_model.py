from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib

X = np.random.randint(0, 100, (100, 7))
y = np.random.randint(0, 2, 100)

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "model.pkl")
