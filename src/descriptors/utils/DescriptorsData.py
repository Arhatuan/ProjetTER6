

class DescriptorsData:
    """Contains the descriptors's data for one singular image"""

    rlm1: list[float]
    """RLM data for object 1 (one value per RLM line)"""
    rlm2: list[float]
    """RLM data for object 2 (one value per RLM line)"""
    forces: dict[float, list[float]]
    """Force data : for each force value, its data is associated to (one value per RLM line)"""


    dist1: list[float]
    """Distance data : from the center point to the object 1 (one value per RLM line)"""
    dist2: list[float]
    """Distance data : from the center point to the object 2 (one value per RLM line)"""

    angles: list[float]
    """Angles data : probabilities of belonging to either "Right", "Left", "Above", "Under" directions (4 values in total)"""


    def __init__(self):
        self.rlm1, self.rlm2 = [], []
        self.forces = dict()
        self.dist1, self.dist2 = [], []
        self.angles = []
