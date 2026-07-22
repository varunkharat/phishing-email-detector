"""Train and compare Logistic Regression, Naive Bayes, and Random Forest classifiers."""

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "naive_bayes": MultinomialNB(),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
}


def train_model(model_name: str, X_train, y_train):
    """Fit the named model on training features/labels and return it."""
    raise NotImplementedError


if __name__ == "__main__":
    pass
