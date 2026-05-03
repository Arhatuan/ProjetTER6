from enum import Enum

class Database(Enum):
    """Compatible databases"""
    S1 = "SimpleShapes1"
    S2 = "SimpleShapes2"
    # ...

class Classifier(Enum):
    """Compatible classifiers"""
    MLP = "Multi-layer Perceptron"
    RF = "Random Forests"
    # ...


class SimpleShapesClasses(Enum):
    """Labels for the directions used in the SimpleShapes's annotations files"""
    CLASSES_4 = ('Above', 'Under', 'Left', 'Right')
    CLASSES_8 = ('Above', 'Under', 'Left', 'Right', 'Above Left', 'Above Right', 'Under Left', 'Under Right')



