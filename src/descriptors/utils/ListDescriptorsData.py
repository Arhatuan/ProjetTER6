from .DescriptorEnum import Descriptor
from .DescriptorsData import DescriptorsData
from .DescriptorsParameters import DescriptorsParameters
from ...utils.utils import get_unique_force_degrees_and_force_descriptors

class ListDescriptorsData:
    """Contains the lists of descriptors's data for multiple images (in a database)"""

    rlm: list
    force: dict[str, list]
    dist1: list
    dist2: list
    angles: list

    def __init__(self):
        self.rlm = []
        self.force = dict()
        self.dist1, self.dist2 = [], []
        self.angles = []

    def add_descriptors_data_from_one_image(self, descriptors_data: DescriptorsData, descriptors_parameters: DescriptorsParameters):
        """Add the descriptors's data from one singular image to the lists of data

        Args:
            descriptors_data (DescriptorsData): descriptors's data from one singular image
            descriptors_parameters (DescriptorsParameters): parameters for the descriptors (in particular the force descriptors)
        """
        self.rlm.append( (descriptors_data.rlm1 + descriptors_data.rlm2))
        self.dist1.append( descriptors_data.dist1 )
        self.dist2.append( descriptors_data.dist2 )
        self.angles.append( descriptors_data.angles )

        for force_descriptor, force_degree in zip(descriptors_parameters.force_descriptors, descriptors_parameters.force_degrees):
            self.force.setdefault(force_descriptor, []).append(descriptors_data.forces[force_degree])

    def generate_X_data_from_descriptors(self, list_tested_descriptors: list[Descriptor | str]) -> list:
        """Generate a list of the descriptors's data (for each image) in the order of the descriptors given as parameters.
        For one image, the descriptors's data are concatenated into a single value

        Args:
            list_tested_descriptors (list[Descriptor | str]): descriptors to compute in order

        Raises:
            ValueError: unsupported descriptor given as parameter

        Returns:
            X_data (list): the ordered list of necessary descriptors's data (per image)
        """
        X_data = []
        force_degrees, force_descriptors = get_unique_force_degrees_and_force_descriptors(list_tested_descriptors)

        # For each image, we add the necessary descriptors (concatenated, in the order given)
        for i in range(len(self.rlm)):
            sum_descriptors = []

            for descriptor in list_tested_descriptors:
                match descriptor:
                    case Descriptor.RLM: sum_descriptors.extend( self.rlm[i] )
                    case Descriptor.DISTANCE: sum_descriptors.extend( self.dist1[i] + self.dist2[i] )
                    case Descriptor.ANGLES: sum_descriptors.extend( self.angles[i] )
                    #case _: raise ValueError(f"Unsupported descriptor (for computing X data) : {descriptor}")

                if descriptor in force_descriptors:
                    sum_descriptors.extend( self.force[descriptor][i] )

            X_data.append(sum_descriptors)
        
        return X_data
