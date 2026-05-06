import os
from pathlib import Path
import time
import csv
from typing import Callable, Any

from ..utils.enumerations import Database, LabelDirection
from ..descriptors import descriptors
from ..descriptors.utils.DescriptorsParameters import DescriptorsParameters
from ..descriptors.utils.ListDescriptorsData import ListDescriptorsData

PATH_DB_SIMPLESHAPES1 = os.path.join(Path(__file__).parent.parent.parent, "images", "SimpleShapes1")
PATH_DB_SIMPLESHAPES2 = os.path.join(Path(__file__).parent.parent.parent, "images", "SimpleShapes2")
PATH_DB_SIG = os.path.join(Path(__file__).parent.parent.parent, "images", "SIG")
PATH_DB_SHARVITSR = os.path.join(Path(__file__).parent.parent.parent, "images", "SharvitSR")

PATH_ANNOTATIONS_SIMPLESHAPES1 = os.path.join(Path(__file__).parent.parent.parent, "annotations", "SimpleShapes1.csv")
PATH_ANNOTATIONS_SIMPLESHAPES2 = os.path.join(Path(__file__).parent.parent.parent, "annotations", "SimpleShapes2.csv")
PATH_ANNOTATIONS_SIG = os.path.join(Path(__file__).parent.parent.parent, "annotations", "SIG.csv")
PATH_ANNOTATIONS_SHARVITSR = os.path.join(Path(__file__).parent.parent.parent, "annotations", "SharvitSR_annotations.csv")

SIMPLESHAPES_BACKGROUND = (68, 1, 84, 255)

class row:
    """Type for a row in the annotations file 
    (used by lambda functions in the main class)"""
    pass

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

    labels4directions: list[str]
    """Possible labels for the ground truth (4 directions)"""
    labels8directions: list[str]
    """Possible labels for the ground truth (8 directions)"""

    conditions: Callable[[row, list[str]], bool]
    """(From a specific row of the annotations file + the list of labels) How to choose if we test this image (must be labeled, not max difficulty...)"""
    getStrImgProcessed: Callable[[row], str]
    """(From a specific row of the annotations file) Gets the name of the image"""
    getGroundTruth: Callable[[row], str]
    """(From a specific row of the annotations file) Gets the ground truth value of the image"""

    def __init__(self, database : Database):
        """Prepare the annotations for the given database ; computing descriptors comes later"""
        self.list_descriptors_data = ListDescriptorsData()
        self.Y_data = []
        self.database = database
        self.__set_path_DB_and_annotations(database)
        self.__set_labels(database)
        self.__set_annotations_parameters(database)

    def __load_annotations(self, csvfile: str):
        """
        Load an annotated csv file (differs from time_management.read_csv(csvfile) -> no values is changed here, and there
        is no 'result' field).

        :param csvfile: csv file to read data.
        :return: list of the rows of the csv file. Rows are defined as dict.
        """
        rows = []
        with open(csvfile, 'r') as file:
            reader = csv.DictReader(file, delimiter=',', quotechar="|")
            for row in reader:
                rows.append(row)
        return rows

    def __set_path_DB_and_annotations(self, database: Database):
        """Automatically sets the path to the database, and loads the annotations, based on the Database given"""
        match database:
            case Database.S1:
                self.path_db = PATH_DB_SIMPLESHAPES1
                self.annotations = self.__load_annotations(PATH_ANNOTATIONS_SIMPLESHAPES1)
            case Database.S2:
                self.path_db = PATH_DB_SIMPLESHAPES2
                self.annotations = self.__load_annotations(PATH_ANNOTATIONS_SIMPLESHAPES2)
            case Database.SIG:
                self.path_db = PATH_DB_SIG
                self.annotations = self.__load_annotations(PATH_ANNOTATIONS_SIG)
            case Database.SHARVITSR:
                self.path_db = PATH_DB_SHARVITSR
                self.annotations = self.__load_annotations(PATH_ANNOTATIONS_SHARVITSR)
            case _: raise ValueError(f"Unsupported database : {database}")

    def __set_labels(self, database: Database):
        """Automatically sets the labels for the 4 directions and 8 directions, that are used in the given database's annotations file"""
        match database:
            case Database.S1 | Database.S2 | Database.SHARVITSR:
                self.labels4directions = LabelDirection.CLASSES_4.value
                self.labels8directions = LabelDirection.CLASSES_8.value
            case Database.SIG:
                # The SIG annotations use lowercase labels. Also, there are only 4 directions tested.
                self.labels4directions = [label.lower() for label in LabelDirection.CLASSES_4.value]
                self.labels8directions = [label.lower() for label in LabelDirection.CLASSES_8.value]
            case _: raise ValueError(f"Unsupported database : {database}")

    def __set_annotations_parameters(self, database: Database):
        """Automatically sets some parameters depending on the annotations file of the given database :
        the conditions to choose the image, the name of the image, the ground truth.
        All these informations are stored differently depending on the annotations file."""
        match database:
            case Database.S1 | Database.S2:
                self.conditions = lambda row, tested_directions: (row['nb'] != "?"
                                                                    #and row['diff'] != '4' # exclude max difficulty
                                                                    and row['rel'] in tested_directions) # is a tested direction
                self.getStrImgProcessed = lambda row: f"img-{row['obj1']}-{row['obj2']}-{row['nb']}.png"
                self.getGroundTruth = lambda row: row["rel"]
            case Database.SIG:
                self.conditions = lambda row, tested_directions: (row['pos'] != "?"
                                                                    #and row['lvl'] != '4' # exclude max difficulty
                                                                    and row['pos'] in tested_directions) # is a tested direction
                self.getStrImgProcessed = lambda row: f"{row['img_name']}"
                self.getGroundTruth = lambda row: row["pos"]
            case Database.SHARVITSR:
                self.conditions = lambda row, tested_directions: (row['card'] != "?"
                                                                    #and row['diff'] != '4' # exclude max difficulty
                                                                    and row['card'] in tested_directions) # is a tested direction
                self.getStrImgProcessed = lambda row: f"img-{row['obj1']}-{row['obj2']}.png"
                self.getGroundTruth = lambda row: row["card"]
            case _: raise ValueError(f"Unsupported database : {database}")


    def compute_descriptors(self, nb_directions: int, descriptors_parameters: DescriptorsParameters):
        """Launch the computing of the descriptors, and print the time taken.

        Args:
            nb_directions (int): the number of directions to test
            descriptors_parameters (DescriptorsParameters): contains the general parameters for the descriptors to compute
        """
        print(f"-> (computing descriptors) For {nb_directions} directions :")
        timeStart = time.time()

        self.__template_compute_descriptors_and_Y_data(nb_directions = nb_directions,
                                                       descriptors_parameters = descriptors_parameters)

        print("\tTime taken : {:.1f}s".format(time.time() - timeStart))


    def __template_compute_descriptors_and_Y_data(self, nb_directions: int, descriptors_parameters: DescriptorsParameters):
        """Compute the descriptors for an image database (the parameters related to that database were assigned at the class's initialization).
        We compute the descriptors that are specified in their parameters, and only for images assigned to the correct number of directions.

        Args:
            nb_directions (int): the number of directions to test
            descriptors_parameters (DescriptorsParameters): the parameters for the descriptors (which descriptors to compute...)

        Raises:
            ValueError: unsupported number of directions
        """
        background = SIMPLESHAPES_BACKGROUND # always the same background for every database...
        total_length = len(self.annotations)

        # 1) Get the directions we are testing (either 4 cardinal directions, or 8)
        match nb_directions:
            case 4: tested_directions = self.labels4directions
            case 8: tested_directions = self.labels8directions
            case _: raise ValueError(f"Unsupported number of directions to test : {nb_directions}")

        for i, row in enumerate(self.annotations):
            # 2) Print the current state of computing the descriptors for each image in the database
            strImgProcessed = self.getStrImgProcessed(row)
            self.__print_current_state_computing_descriptors(i+1, total_length, strImgProcessed)
            
            # 3) If the image meets some conditions, we test it (see conditions in function '__set_annotations_parameters')
            if (self.conditions(row, tested_directions)):
                #img_name = os.path.join(self.path_db, strImgProcessed)
                img_name = f"{self.path_db}/{strImgProcessed}"

                # 4) Compute the descriptors
                descriptors_data = descriptors.image_processing_v4(imagename             = img_name,
                                                                    background            = background,
                                                                    step                  = 360 / descriptors_parameters.nb_radial_lines,
                                                                    descriptorsParameters = descriptors_parameters)
                # 5) Save the values for each descriptor, for each image
                self.list_descriptors_data.add_descriptors_data_from_one_image(descriptors_data, descriptors_parameters)
                # 6) Save the ground truth for each image
                self.Y_data.append( self.getGroundTruth(row) )


    def __print_current_state_computing_descriptors(self, nb: int, total_length: int, img_name: str):
        """Print current progress of computing descriptors (in percentage of images dealt with)"""
        strProgress = "(progress: {:2.1%})".format(nb/total_length)
        print("{:<16} {:>18}".format(img_name, strProgress), end="\r")
        if nb == total_length: print() # to cancel the last carriage return character '\r'
