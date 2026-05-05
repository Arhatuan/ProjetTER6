from enum import Enum
import numpy as np

from .utils.Parameters import Parameters
from .descriptors.utils.DescriptorsParameters import DescriptorsParameters
from .dataComputing.ComputeDescriptorsFromDatabase import ComputeDescriptorsFromDatabase
from .dataComputing.ComputeResults import ComputeResults
from .dataComputing.SaveAndDisplayResults import SaveAndDisplayResults
from .utils.save_load_computed_descriptors import save_computed_descriptors, load_computed_descriptors

class ManagerOptions(Enum):
    DEFAULT = "default"
    SAVE = "save"
    LOAD = "load"


def manage(parameters: Parameters, option: ManagerOptions):
    """Assign the process corresponding to the given option

    Args:
        parameters (Parameters): the parameters to share to the next process
        option (ManagerOptions): option for the process to apply

    Raises:
        ValueError: unsupported option
    """
    match option:
        case ManagerOptions.DEFAULT: _compute_descriptors_and_results(parameters)
        case ManagerOptions.SAVE: _compute_and_save_descriptors(parameters)
        case ManagerOptions.LOAD: _load_descriptors_values_and_compute_results(parameters)
        case _: raise ValueError(f"Unsupported manager option : {option}")



def _compute_descriptors_and_results(parameters: Parameters):
    """DEFAULT process : compute some descriptors, then train models and compute results, 
    before showing them in the terminal and saving them in a file.

    Args:
        parameters (Parameters): parameters for the program (obtained from the command line)
    """
    _show_parameters(parameters)

    global_results = SaveAndDisplayResults()

    # 1) Prepare the parameters for the descriptors (which to compute...)
    descriptors_parameters = DescriptorsParameters(parameters)

    for nb_directions in parameters.nb_directions:
        # 2) Prepare the annotations's file for the given database
        descriptors_computing = ComputeDescriptorsFromDatabase(parameters.database)

        # 3) Compute the descriptors
        descriptors_computing.compute_descriptors(nb_directions, descriptors_parameters)

        # 4) Compute the results
        results_computing = ComputeResults()
        results_computing.compute_results_from_descriptors_combinations_and_classifiers(
            nb_directions       = nb_directions,
            computedDescriptors = descriptors_computing,
            parameters          = parameters
        )

        # 5) Add the results to the global results
        global_results.insert_results(nb_directions, results_computing)

    # 6) Display the results
    global_results.display_results()
    global_results.write_results_in_file(parameters.database)



def _compute_and_save_descriptors(parameters: Parameters):
    """SAVE process : compute some descriptors, and save them in a file. Do not compute results.

    Args:
        parameters (Parameters): parameters for the program (obtained from the command line)
    """

    # 0) Simplify the parameters (keep only what is necessary to save)
    parameters.classifiers = [] # no classifier

    # descriptors : show the unique descriptors, not the combinations
    descriptor_name = lambda descriptor: descriptor if type(descriptor) is str else descriptor.value
    parameters.descriptors_layout = [ sorted(parameters.get_unique_descriptors(), key=descriptor_name) ] 

    # nb directions is only the first one
    nb_directions = parameters.nb_directions[0]
    parameters.nb_directions = [nb_directions]

    _show_parameters(parameters)
    print("== Operation : SAVE descriptors ==\n")


    # 1) Compute the descriptors
    descriptors_parameters = DescriptorsParameters(parameters)
    descriptors_computing = ComputeDescriptorsFromDatabase(parameters.database)
    descriptors_computing.compute_descriptors(nb_directions, descriptors_parameters)

    # 2) Save the objects
    filename = save_computed_descriptors(parameters, nb_directions, descriptors_computing)
    print(f"Computed descriptors saved in the file '{filename}'")



def _load_descriptors_values_and_compute_results(new_parameters: Parameters):
    params, nb_directions, computedDescriptors = load_computed_descriptors(filename = new_parameters.filename)

    raise NotImplementedError("Load process not yet implemented")

    print(len(computedDescriptors.Y_data))
    print(len(computedDescriptors.annotations))
    new_annotations = [row for row in computedDescriptors.annotations if computedDescriptors.conditions(row, computedDescriptors.labels4directions)]
    print(len(new_annotations))
    #print(np.where(computedDescriptors.conditions(computedDescriptors.annotations, computedDescriptors.labels4directions)))
    indexes_new_annotations = [i for i, row in enumerate(computedDescriptors.annotations) if computedDescriptors.conditions(row, computedDescriptors.labels4directions)]
    print(len(indexes_new_annotations))
    print(indexes_new_annotations)
    pass




def _show_parameters(parameters: Parameters):
    """Show the parameters, so that we know which ones will be computed

    Args:
        parameters (Parameters): parameters inside a `Parameters` instance
    """
    print("Parameters :")
    print(f"\t• Database : {parameters.database.value}")
    print("\t• Classifier(s) : {}".format( ", ".join([c.value for c in parameters.classifiers]) ))
    print("\t• Nb of directions : {}".format( ", ".join([str(nb_dir) for nb_dir in parameters.nb_directions]) ))
   
    # To print descriptors combinations properly
    get_str_descriptor = lambda descriptor : descriptor if type(descriptor) is str else descriptor.value
    get_str_combination = lambda list_descriptors : [get_str_descriptor(descriptor) for descriptor in list_descriptors]
    sub_str_list = ", ".join([str(get_str_combination(sub_list)) for sub_list in parameters.descriptors_layout])
    print("\t• Descriptors : {}".format( str(sub_str_list) ))
    print()

