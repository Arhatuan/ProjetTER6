import random

from ...utils.Parameters import Parameters
from .DescriptorEnum import Descriptor
from ...utils.utils import get_unique_force_degrees_and_force_descriptors
from ...utils.enumerations import ReferencePointMode

class DescriptorsParameters:
    """Contains the parameters for the descriptors : which descriptors must be computed (we don't necessarily want to compute every descriptors)"""

    force_degrees: list[float]
    """The list of different force values to test (2, 0, 1.3, -2.7, ...)"""
    force_descriptors: list[str]
    """The list of different force descriptors (f2, f0, f1.3, f-2.7, ...)"""

    computeRLM: bool
    """Flag for computing RLM data, or not"""
    computeDist: bool
    """Flag for computing Dist data, or not"""
    computeAngles: bool
    """Flag for computing Angles data (4 directions), or not"""
    computeAngles8: bool
    """Flag for computing Angles data (8 directions), or not"""

    nb_radial_lines: int
    """The number of radial lines"""
    rp_mode: ReferencePointMode
    """How the point Rp is selected."""
    rp_border_size: int
    """Border size used by random border/center Rp modes."""
    rp_selected_point: tuple[int, int] | None
    """Selected Rp for the current iteration (absolute image coordinates)."""
    rp_rng: random.Random | None
    """Random generator bound to the current iteration."""
    

    def __init__(self, parameters: Parameters):
        unique_descriptors = parameters.get_unique_descriptors()
        self.force_degrees, self.force_descriptors = get_unique_force_degrees_and_force_descriptors(unique_descriptors)
        
        self.computeRLM = Descriptor.RLM in unique_descriptors
        self.computeDist = Descriptor.DISTANCE in unique_descriptors
        self.computeAngles = Descriptor.ANGLES in unique_descriptors or Descriptor.ANGLES8 in unique_descriptors
        self.computeAngles8 = Descriptor.ANGLES8 in unique_descriptors

        self.nb_radial_lines = parameters.nb_radial_lines
        self.rp_mode = parameters.rp_mode
        self.rp_border_size = parameters.rp_border_size
        self.rp_selected_point = None
        self.rp_rng = None

    def reset_rp_for_iteration(self, seed: int):
        """Reset Rp random state for one training iteration."""
        self.rp_selected_point = None
        self.rp_rng = random.Random(seed)
