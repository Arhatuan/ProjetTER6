from enum import Enum

class Database(Enum):
    """Compatible databases"""
    S1 = "SimpleShapes1"
    S2 = "SimpleShapes2"
    SIG = "SIG"
    SHARVITSR = "SharvitSR"
    # ...

class Classifier(Enum):
    """Compatible classifiers"""
    MLP = "Multi-layer Perceptron"
    RF = "Random Forests"
    # ...

class ReferencePointMode(Enum):
    """How the point Rp is selected."""
    DETERMINISTIC = "deterministic"
    RANDOM_FULL = "random_full"
    RANDOM_BORDER = "random_border"
    RANDOM_CENTER = "random_center"


class LabelDirection(Enum):
    """Labels for the directions used in a similar form in different annotations files"""
    CLASSES_4 = ('Above', 'Under', 'Left', 'Right')
    CLASSES_8 = ('Above', 'Under', 'Left', 'Right', 'Above Left', 'Above Right', 'Under Left', 'Under Right')

