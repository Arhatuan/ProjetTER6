from enum import Enum

class Descriptor(Enum):
    """Compatible descriptors"""
    RLM = "rlm"
    # Force can be "f2", "f0", "f1.3"... 
    #   So it is taken separately
    DISTANCE = "distance"
    ANGLES = "angles"

FORCE_REGEX = r"^f(-?\d+(\.\d+)?)$"
"""The regex to recognize a force ('f2', 'f0', 'f1.3', 'f-0.5', ... )"""