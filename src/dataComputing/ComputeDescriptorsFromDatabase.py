import os
from pathlib import Path
import time

from ..utils.enumerations import Database, SimpleShapesClasses
from ..extendedRLM.model_learning import load_annotations
from ..descriptors import descriptors
from ..descriptors.utils.DescriptorsParameters import DescriptorsParameters
from ..descriptors.utils.ListDescriptorsData import ListDescriptorsData

PATH_DB_SIMPLESHAPES1 = os.path.join(Path(__file__).parent.parent.parent, "images", "SimpleShapes1")
PATH_DB_SIMPLESHAPES2 = os.path.join(Path(__file__).parent.parent.parent, "images", "SimpleShapes2")

PATH_ANNOTATIONS_SIMPLESHAPES1 = os.path.join(Path(__file__).parent.parent.parent, "annotations", "SimpleShapes1.csv")
PATH_ANNOTATIONS_SIMPLESHAPES2 = os.path.join(Path(__file__).parent.parent.parent, "annotations", "SimpleShapes2.csv")

SIMPLESHAPES_BACKGROUND = (68, 1, 84, 255)

class ComputeDescriptorsFromDatabase:
    """Compute the descriptors for every image in a given image database"""

    list_descriptors_data: ListDescriptorsData
    """Descriptors's data (one per image in the database)"""
    Y_data: list
    """Ground truth (one per image)"""

    database: Database
    path_db: str
    annotations: list
    """The annotations are the list of data from the annotations file (each row corresponds to an image)"""

    def __init__(self, database : Database):
        """Prepare the annotations for the given database ; computing descriptors comes later"""
        self.list_descriptors_data = ListDescriptorsData()
        self.Y_data = []
        self.database = database
        self.__set_path_DB_and_annotations(database)

    def __set_path_DB_and_annotations(self, database: Database):
        """Automatically sets the path to the database, and loads the annotations, based on the Database given"""
        match database:
            case Database.S1:
                self.path_db = PATH_DB_SIMPLESHAPES1
                self.annotations = load_annotations(PATH_ANNOTATIONS_SIMPLESHAPES1)
            case Database.S2:
                self.path_db = PATH_DB_SIMPLESHAPES2
                self.annotations = load_annotations(PATH_ANNOTATIONS_SIMPLESHAPES2)
            case _: raise ValueError(f"Unsupported database : {database}")

    def compute_descriptors(self, nb_directions: int, descriptors_parameters: DescriptorsParameters):
        """Redirects the descriptors's computing to the function that manages the given database

        Args:
            nb_directions (int): the number of directions to test
            descriptors_parameters (DescriptorsParameters): contains the general parameters for the descriptors to compute
        """
        print(f"-> (computing descriptors) For {nb_directions} directions :")
        timeStart = time.time()

        match self.database:
            case Database.S1 | Database.S2:
                self.__compute_descriptors_and_Y_data_on_SimpleShape(nb_directions = nb_directions,
                                                                     descriptors_parameters = descriptors_parameters)
            case _: raise ValueError(f"Unsupported database : {database}")

        print("\tTime taken : {:.1f}s".format(time.time() - timeStart))


    def __compute_descriptors_and_Y_data_on_SimpleShape(self, nb_directions: int, descriptors_parameters: DescriptorsParameters):
        """Compute the descriptors from SimpleShapes 1 or 2, based on the number of directions and some other parameters

        Args:
            nb_directions (int): the number of directions to test
            descriptors_parameters (DescriptorsParameters): the parameters for the descriptors (which descriptors to compute...)

        Raises:
            ValueError: unsupported number of directions
        """
        background = SIMPLESHAPES_BACKGROUND
        total_length = len(self.annotations)

        # 1) Get the directions we are testing (either 4 cardinal directions, or 8)
        match nb_directions:
            case 4: tested_directions = SimpleShapesClasses.CLASSES_4.value
            case 8: tested_directions = SimpleShapesClasses.CLASSES_8.value
            case _: raise ValueError(f"Unsupported number of directions to test : {nb_directions}")

        # 2) Conditions on the images to test (not every image is tested)
        conditions = lambda row: (row['nb'] != "?"
                                    #and row['diff'] != '4' # exclude max difficulty
                                    and row['rel'] in tested_directions) # is a tested direction

        for i, row in enumerate(self.annotations):
            # 3) Print the current state of computing the descriptors for each image in the database
            strImgProcessed = f"img-{row['obj1']}-{row['obj2']}-{row['nb']}.png"
            self.__print_current_state_computing_descriptors(i+1, total_length, strImgProcessed)

            if (conditions(row)):
                img_name = f"{self.path_db}/img-{row['obj1']}-{row['obj2']}-{row['nb']}.png"

                # 4) Compute the descriptors
                descriptors_data = descriptors.image_processing_v4(imagename             = img_name,
                                                                   background            = background,
                                                                   step                  = 360 / descriptors_parameters.nb_radial_lines,
                                                                   descriptorsParameters = descriptors_parameters)
                # 5) Save the values for each descriptor, for each image
                self.list_descriptors_data.add_descriptors_data_from_one_image(descriptors_data, descriptors_parameters)
                # 6) Save the ground truth for each image
                self.Y_data.append(row["rel"])


    def __print_current_state_computing_descriptors(self, nb: int, total_length: int, img_name: str):
        """Print current progress of computing descriptors (in percentage of images dealt with)"""
        strProgress = "(progress: {:2.1%})".format(nb/total_length)
        print("{:<16} {:>18}".format(img_name, strProgress), end="\r")
        if nb == total_length: print() # to cancel the last carriage return character '\r'

    