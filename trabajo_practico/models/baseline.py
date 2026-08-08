from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class BaselineClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y=None):
        self.classes_ = np.array(["not satisfied", "satisfied"])
        return self

    def compute_baseline(self, row):
        baggage = 1 if row["Baggage handling"] > 3 else 0
        seat = 1 if row["Seat comfort"] > 3 else 0
        short_flight = 1 if row["Flight Distance"] < 1000 else 0
        is_business = 1 if row["Type of Travel"] == "Business travel" else 0
        on_time = 0 if row["Arrival Delay in Minutes"] >= 15 else 1
        class_type = 1 if row["Class"] == "Business" else 0

        index = (
            (baggage + on_time) * 0.2
            + 0.1 * (seat + is_business + short_flight)
            + class_type * 0.3
        )
        if index >= 0.5:
            return "satisfied"
        return "not satisfied"

    def predict(self, X):
        return X.apply(self.compute_baseline, axis=1).values
