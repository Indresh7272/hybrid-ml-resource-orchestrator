from functools import lru_cache
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

@lru_cache(maxsize=1)
def get_model():
    data = load_iris()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(data.data, data.target)
    return model, data.target_names.tolist()

def predict_iris(features: list[float]) -> dict:
    model, target_names = get_model()
    predicted_class = int(model.predict([features])[0])
    probabilities = model.predict_proba([features])[0]
    return {
        "predicted_class_id": predicted_class,
        "predicted_class": target_names[predicted_class],
        "confidence": round(float(max(probabilities)), 4)
    }
