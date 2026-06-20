from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import nll, softmax


@dataclass
class TemperatureScaler:
    temperature: float = 1.0
    min_temperature: float = 0.05
    max_temperature: float = 20.0

    def fit(self, logits, labels) -> "TemperatureScaler":
        logits = self._validate_logits(logits)
        labels = np.asarray(labels, dtype=int)
        if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
            raise ValueError("labels must be a 1D array with the same length as logits")

        log_min = float(np.log(self.min_temperature))
        log_max = float(np.log(self.max_temperature))
        self.temperature = float(np.exp(self._golden_section_search(logits, labels, log_min, log_max)))
        return self

    def transform_logits(self, logits) -> np.ndarray:
        logits = self._validate_logits(logits)
        return logits / float(self.temperature)

    def predict_proba(self, logits) -> np.ndarray:
        return softmax(self.transform_logits(logits))

    def predict_positive_proba(self, logits) -> np.ndarray:
        return self.predict_proba(logits)[:, 1]

    def to_dict(self) -> dict[str, float]:
        return {"temperature": float(self.temperature)}

    @classmethod
    def from_dict(cls, data: dict) -> "TemperatureScaler":
        return cls(temperature=float(data["temperature"]))

    @staticmethod
    def _validate_logits(logits) -> np.ndarray:
        values = np.asarray(logits, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError(f"logits must have shape (n, 2), got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("logits contain non-finite values")
        return values

    @staticmethod
    def _golden_section_search(logits: np.ndarray, labels: np.ndarray, low: float, high: float) -> float:
        ratio = (np.sqrt(5.0) - 1.0) / 2.0
        x1 = high - ratio * (high - low)
        x2 = low + ratio * (high - low)

        def objective(log_temperature: float) -> float:
            temperature = float(np.exp(log_temperature))
            probs = softmax(logits / temperature)[:, 1]
            return nll(labels, probs)

        f1 = objective(x1)
        f2 = objective(x2)
        for _ in range(100):
            if abs(high - low) < 1e-6:
                break
            if f1 > f2:
                low = x1
                x1 = x2
                f1 = f2
                x2 = low + ratio * (high - low)
                f2 = objective(x2)
            else:
                high = x2
                x2 = x1
                f2 = f1
                x1 = high - ratio * (high - low)
                f1 = objective(x1)
        return float((low + high) / 2.0)
