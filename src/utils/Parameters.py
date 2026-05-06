from .enumerations import Database, Classifier, ReferencePointMode
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
    rp_mode: ReferencePointMode
    """How the point Rp is chosen."""
    rp_border_size: int
    """Border size (in pixels) used by random border/center Rp modes."""
    nb_iterations: int
    """Number of independent training iterations."""
    random_seed: int
    """Base random seed for independent runs."""

    def __init__(self):
        self.database = None
        self.classifiers = []
        self.descriptors_layout = []
        self.nb_directions = []
        self.nb_radial_lines = 120
        self.rp_mode = ReferencePointMode.DETERMINISTIC
        self.rp_border_size = 20
        self.nb_iterations = 1
        self.random_seed = 42


    # SETTERS
    def set_database(self, database: Database):
        self.database = database

    def add_classifier(self, classifier: Classifier):
        self.classifiers.append(classifier)

    def add_descriptors_combination(self, descriptors_combination: list[Descriptor | str]):
        self.descriptors_layout.append(descriptors_combination)

    def add_nb_directions(self, nb_directions: int):
        self.nb_directions.append(nb_directions)

    def set_rp_mode(self, rp_mode: ReferencePointMode):
        self.rp_mode = rp_mode

    def set_rp_border_size(self, rp_border_size: int):
        self.rp_border_size = rp_border_size

    def set_nb_iterations(self, nb_iterations: int):
        self.nb_iterations = nb_iterations

    def set_random_seed(self, random_seed: int):
        self.random_seed = random_seed


    # OTHERS
    def get_unique_descriptors(self) -> list[Descriptor | str]:
        """Return the list of unique different descriptors

        Returns:
            list[Descriptor | str]: the list of unique different descriptors (each different force value is also distinct)
        """
        return list(set([descriptor for descriptors_list in self.descriptors_layout for descriptor in descriptors_list]))
