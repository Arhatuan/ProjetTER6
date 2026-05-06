import dill
import os
from pathlib import Path

from .Parameters import Parameters
from .utils import get_key_descriptors_combination
from ..dataComputing.ComputeDescriptorsFromDatabase import ComputeDescriptorsFromDatabase

PATH_SAVED_DESCRIPTORS_DIR = os.path.join(Path(__file__).parent.parent.parent, "savedDescriptors")

def save_computed_descriptors(params: Parameters, nb_directions: int, computedDescriptors: ComputeDescriptorsFromDatabase) -> str:
    """Save some data to the 'savedDescriptors' folder

    Args:
        params (Parameters): the parameters
        nb_directions (int): the number of directions
        computedDescriptors (ComputeDescriptorsFromDatabase): the computed descriptors

    Returns:
        str: the filename where the data was saved
    """

    data = {
        "parameters": params,
        "nb_directions": nb_directions,
        "computed_descriptors": computedDescriptors
    }
    
    # Choose the file name
    db_name = computedDescriptors.database.name[:7]
    descriptors = get_key_descriptors_combination(params.descriptors_layout[0])
    filename = f"{db_name}_{nb_directions}directions_{descriptors}.pkl"

    # Create the directory if necessary
    if not os.path.isdir(PATH_SAVED_DESCRIPTORS_DIR):
        os.mkdir(PATH_SAVED_DESCRIPTORS_DIR)

    # Save the file
    full_filename = f"{PATH_SAVED_DESCRIPTORS_DIR}/{filename}"
    with open(full_filename, "wb") as f:
        dill.dump(data, f)

    return f"savedDescriptors/{filename}"

def load_computed_descriptors(filename: str) -> tuple[Parameters, int, ComputeDescriptorsFromDatabase]:
    """Load some data from the 'savedDescriptors' folder

    Args:
        filename (str): the name of the file to load

    Returns:
        tuple[Parameters, int, ComputeDescriptorsFromDatabase]: the parameters, the number of directions and the computed descriptors that were saved in the file
    """
    full_filename = f"{PATH_SAVED_DESCRIPTORS_DIR}/{filename}"

    with open(full_filename, "rb") as f:
        data = dill.load(f)

    params = data["parameters"]
    nb_directions = data["nb_directions"]
    computedDescriptors = data["computed_descriptors"]

    return (params, nb_directions, computedDescriptors)
