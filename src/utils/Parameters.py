from .enumerations import Database, Classifier
from ..descriptors.utils.DescriptorEnum import Descriptor

class Parameters:
    """General parameters for the program"""

    database: Database
    """The image database to work on (only one)"""
    classifiers: list[Classifier]
    """The classifiers to work with (multiple ones possible, tested separately)"""
    descriptors_layout: list[list[Descriptor | str]]
    """Descriptors's combinations (multiple combinations possible, tested separately)"""
    nb_directions: list[int]
    """Number of directions to work on : 4 or 8 (either one, or the two values ; tested separately)"""

    nb_radial_lines: int
    """The number of radial lines"""

    def __init__(self):
        self.database = None
        self.classifiers = []
        self.descriptors_layout = []
        self.nb_directions = []
        self.nb_radial_lines = 120


    # SETTERS
    def set_database(self, database: Database):
        self.database = database

    def add_classifier(self, classifier: Classifier):
        self.classifiers.append(classifier)

    def add_descriptors_combination(self, descriptors_combination: list[Descriptor | str]):
        self.descriptors_layout.append(descriptors_combination)

    def add_nb_directions(self, nb_directions: int):
        self.nb_directions.append(nb_directions)


    # OTHERS
    def get_unique_descriptors(self) -> list[Descriptor | str]:
        """Return the list of unique different descriptors

        Returns:
            list[Descriptor | str]: the list of unique different descriptors (each different force value is also distinct)
        """
        return list(set([descriptor for descriptors_list in self.descriptors_layout for descriptor in descriptors_list]))

