from enum import Enum

class Database(Enum):
    """Compatible databases"""
    S1 = "SimpleShapes1"
    S2 = "SimpleShapes2"
    SIG = "SIG"
    SHARVITSR = "SharvitSR"
    S1andS2 = "S1+S2"
    # ...

class Classifier(Enum):
    """Compatible classifiers"""
    MLP = "Multi-layer Perceptron"
    RF = "Random Forests"
    # ...


class LabelDirection(Enum):
    """Labels for the directions used in a similar form in different annotations files"""
    CLASSES_4 = ('Above', 'Under', 'Left', 'Right')
    CLASSES_8 = ('Above', 'Under', 'Left', 'Right', 'Above Left', 'Above Right', 'Under Left', 'Under Right')


