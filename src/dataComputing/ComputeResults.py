from collections import Counter
import time

import numpy as np
from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.neural_network import MLPClassifier

from ..utils.Parameters import Parameters
from ..utils.enumerations import Classifier
from ..utils.utils import get_key_descriptors_combination
from .ComputeDescriptorsFromDatabase import ComputeDescriptorsFromDatabase


class ComputeResults:
    scores: dict[str, dict[Classifier, list[float]]]
    """Scores per descriptors's combination, and then per classifier."""
    matrices: dict[str, dict[Classifier, ndarray]]
    """Confusion matrices per descriptors's combination, and then per classifier."""
    predictions: dict[str, dict[Classifier, ndarray]]
    """Predictions per descriptors's combination and per classifier."""
    metrics: dict[str, dict[Classifier, dict[str, float]]]
    """Metrics (accuracy, precision, recall, f1) per descriptors's combination and per classifier."""

    def __init__(self):
        self.scores = dict()
        self.matrices = dict()
        self.predictions = dict()
        self.metrics = dict()

    def __build_model(self, classifier: Classifier, random_seed: int):
        match classifier:
            case Classifier.MLP:
                return MLPClassifier(
                    hidden_layer_sizes=(4, 448),
                    solver="adam",
                    max_iter=1000,
                    random_state=random_seed,
                )
            case Classifier.RF:
                return RandomForestClassifier(random_state=random_seed)
            case _:
                raise ValueError(f"Unsupported classifier : {classifier}")

    def __build_cross_validator(self, Y: list, random_seed: int) -> StratifiedKFold:
        class_counts = Counter(Y)
        min_count = min(class_counts.values()) if len(class_counts) > 0 else 0
        n_splits = min(5, min_count)
        if n_splits < 2:
            raise ValueError(
                f"Not enough samples per class for cross-validation (minimum class count={min_count})"
            )
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    def __compute_metrics(self, Y_true: list, Y_pred: ndarray) -> dict[str, float]:
        return {
            "accuracy": float(accuracy_score(Y_true, Y_pred)),
            "precision_macro": float(precision_score(Y_true, Y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(Y_true, Y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(Y_true, Y_pred, average="macro", zero_division=0)),
        }

    def compute_results_from_descriptors_combinations_and_classifiers(
        self,
        nb_directions: int,
        computedDescriptors: ComputeDescriptorsFromDatabase,
        parameters: Parameters,
        random_seed: int = 42,
    ):
        total_nb_descriptors_combinations = len(parameters.descriptors_layout) * len(parameters.classifiers)
        print(f"-> (computing results) For {nb_directions} directions :")
        timeStart = time.time()

        match nb_directions:
            case 4:
                labels = computedDescriptors.labels4directions
            case 8:
                labels = computedDescriptors.labels8directions
            case _:
                raise ValueError(f"Unsuported number of directions for labels : {nb_directions}")

        for i, descriptors_combination in enumerate(parameters.descriptors_layout):
            X_data = computedDescriptors.list_descriptors_data.generate_X_data_from_descriptors(descriptors_combination)

            key_descriptors_combination = get_key_descriptors_combination(descriptors_combination)
            self.scores[key_descriptors_combination] = dict()
            self.matrices[key_descriptors_combination] = dict()
            self.predictions[key_descriptors_combination] = dict()
            self.metrics[key_descriptors_combination] = dict()

            for j, classifier in enumerate(parameters.classifiers):
                current_nb = i * len(parameters.classifiers) + j + 1
                self.__print_current_state_computing_results(current_nb, total_nb_descriptors_combinations)

                try:
                    clf = self.__build_model(classifier, random_seed=random_seed)
                    cv = self.__build_cross_validator(computedDescriptors.Y_data, random_seed=random_seed)

                    scores = cross_val_score(clf, X_data, computedDescriptors.Y_data, cv=cv)
                    Y_predictions = cross_val_predict(clf, X_data, computedDescriptors.Y_data, cv=cv)

                    self.scores[key_descriptors_combination][classifier] = scores
                    self.matrices[key_descriptors_combination][classifier] = confusion_matrix(
                        computedDescriptors.Y_data, Y_predictions, labels=labels
                    )
                    self.predictions[key_descriptors_combination][classifier] = np.array(Y_predictions)
                    self.metrics[key_descriptors_combination][classifier] = self.__compute_metrics(
                        computedDescriptors.Y_data, Y_predictions
                    )

                except Exception as e:
                    print(
                        f"\n\tError when attempting to train classifier {classifier.value} on data from descriptors {key_descriptors_combination} : {e}"
                    )
                    self.scores[key_descriptors_combination][classifier] = np.zeros(5, dtype="float32")
                    self.matrices[key_descriptors_combination][classifier] = np.zeros(
                        (len(labels), len(labels)), dtype="int32"
                    )
                    default_prediction = labels[0] if len(labels) > 0 else ""
                    self.predictions[key_descriptors_combination][classifier] = np.array(
                        [default_prediction for _ in computedDescriptors.Y_data]
                    )
                    self.metrics[key_descriptors_combination][classifier] = {
                        "accuracy": 0.0,
                        "precision_macro": 0.0,
                        "recall_macro": 0.0,
                        "f1_macro": 0.0,
                    }

        print("\tTime taken : {:.1f}s\n".format(time.time() - timeStart))

    def __print_current_state_computing_results(self, nb: int, total_nb_combinations: int):
        strCurrent = f"Descriptors combination {nb} "
        strProgress = "(progress: {:2.1%})".format(nb / total_nb_combinations)
        print("{:<27} {:>18}".format(strCurrent, strProgress), end="\r")
        if nb == total_nb_combinations:
            print()
