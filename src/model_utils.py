from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def build_classifier(name: str):
    if name == "logreg":
        return LogisticRegression(max_iter=2000)
    if name == "svm":
        return LinearSVC()
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=42)
    raise ValueError(f"Unsupported classifier: {name}")
