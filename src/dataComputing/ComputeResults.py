from ..utils.enumerations import Classifier, SimpleShapesClasses
from .ComputeDescriptorsFromDatabase import ComputeDescriptorsFromDatabase
from ..utils.Parameters import Parameters
from ..utils.utils import get_key_descriptors_combination

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
import numpy as np
from numpy import ndarray
import time

class ComputeResults:

    scores: dict[str, dict[Classifier, list[float]]]
    """Scores per descriptors's combination, and then per classifier. Scores are a list of 5 probabilities (from cross-validation on 5 subsets)."""
    matrices: dict[str, dict[Classifier, ndarray]]
    """Confusion matrices per descriptors's combination, and then per classifier. Associated to the scores accessed with the same keys."""
    
    def __init__(self):
        self.scores = dict()
        self.matrices = dict()

    def __get_trained_model(self, X: list, Y: list, classifier: Classifier):
        """Train a model given data (X) and the corresponding ground truth (Y)

        Args:
            X (list): data to learn from. They are values obtained by descriptors.
            Y (list): the ground truth of the X data (Y[i] correspondonds to data X[i]).
            classifier (Classifier): the classifier to train.

        Raises:
            ValueError: the classifier is unsupported.

        Returns:
            classifier: the trained classifier.
        """
        # 1) We train the classifier on 4/5 of the data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        # 2) Choose the classifier
        match classifier:
            case Classifier.MLP:    clf = MLPClassifier(hidden_layer_sizes=(4, 448), solver='adam', max_iter=1000)
            case Classifier.RF:     clf = RandomForestClassifier()
            case _:                 raise ValueError(f"Unsupported classifier : {classifier}")
        
        # 3) Train the classifier
        clf.fit(X_train, Y_train)
        return clf
    
    def __get_scores(self, clf, X: list, Y: list, printScores = False, padding = ""):
        """Get the scores of the classifier on the given data.
        It does a cross-validation on 5 subsets of the data.

        Args:
            clf: the classifier to test.
            X (list): data to test. They are values obtained by descriptors.
            Y (list): the ground truth of the X data (Y[i] corresponds to the data X[i]).
            printScores (bool, optional): print the scores. Defaults to False.
            padding (str, optional): padding before printing the scores. Defaults to "".

        Returns:
            list: 5 scores, one for each of the 5 subsets of the data. 
        """
        # Cross-validation on 5 subsets
        scores = cross_val_score(clf, X, Y, cv=5)
        if printScores:
            print(padding + "%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
            #print("Results per sub-base in cross-validation : \n", scores)
        return scores


    def compute_results_from_descriptors_combinations_and_classifiers(self, nb_directions: int, computedDescriptors: ComputeDescriptorsFromDatabase, parameters: Parameters):
        total_nb_descriptors_combinations = len(parameters.descriptors_layout) * len(parameters.classifiers)
        print(f"-> (computing results) For {nb_directions} directions :")
        timeStart = time.time()

        match nb_directions:
            case 4: labels = SimpleShapesClasses.CLASSES_4.value
            case 8: labels = SimpleShapesClasses.CLASSES_8.value
            case _: raise ValueError(f"Unsuported number of directions for labels : {nb_directions}")

        for i, descriptors_combination in enumerate(parameters.descriptors_layout):
            # Get X_data for a specific combination of descriptors
            X_data = computedDescriptors.list_descriptors_data.generate_X_data_from_descriptors(descriptors_combination)
            
            # Create a key based on the combination : 'rlm+f2+angles+f-2.7' for example
            key_descriptors_combination = get_key_descriptors_combination(descriptors_combination)
            self.scores[key_descriptors_combination] = dict() 
            self.matrices[key_descriptors_combination] = dict()

            for j, classifier in enumerate(parameters.classifiers):
                self.__print_current_state_computing_results(i*2+j+1, total_nb_descriptors_combinations)
                
                try:
                    # 1) Get a trained model
                    clf = self.__get_trained_model(X_data, computedDescriptors.Y_data, classifier)
                    
                    # 2) Compute scores (from cross-validation on 5 subsets -> 1 score for each subset)
                    scores = self.__get_scores(clf, X_data, computedDescriptors.Y_data, printScores=False)
                    self.scores[key_descriptors_combination][classifier] = scores

                    # 3) Compute the confusion matrix
                    Y_predictions = clf.predict(X_data)
                    conf_matrix = confusion_matrix(computedDescriptors.Y_data, Y_predictions, labels=labels)
                    self.matrices[key_descriptors_combination][classifier] = conf_matrix
                
                except Exception as e:
                    # Error when training a model -> default to result = 0.00
                    print(f"\n\tError when attempting to train classifier {classifier.value} on data from descriptors {key_descriptors_combination} : {e}")
                    self.scores[key_descriptors_combination][classifier] = np.zeros(5, dtype="float32")
                    self.matrices[key_descriptors_combination][classifier] = np.zeros((len(labels), len(labels)), dtype='int32')
        
        print("\tTime taken : {:.1f}s\n".format(time.time() - timeStart))


    def __print_current_state_computing_results(self, nb: int, total_nb_combinations: int):
        strCurrent = f"Descriptors combination {nb} "
        strProgress = "(progress: {:2.1%})".format(nb/total_nb_combinations)
        print("{:<27} {:>18}".format(strCurrent, strProgress), end="\r")
        if nb == total_nb_combinations: print() # to cancel the last carriage return character '\r'
