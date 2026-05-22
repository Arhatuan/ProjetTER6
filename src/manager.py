from enum import Enum
import numpy as np

from .utils.Parameters import Parameters
from .utils.enumerations import Database
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

    # Option) Compute SIG descriptors if necessary as a test base
    if parameters.compute_SIG_test_results:
        SIG_computed_descriptors = ComputeDescriptorsFromDatabase(Database.SIG)
        SIG_computed_descriptors.compute_descriptors(4, descriptors_parameters)
        parameters.SIG_computed_descriptors = SIG_computed_descriptors
        

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
    """LOAD process : load a file containing descriptors data, and use them for computing results based on wanted descriptors combinations and classifiers.

    Args:
        new_parameters (Parameters): the new parameters which we want to test on the loaded descriptors data

    Raises:
        ValueError: some of the descriptors to test were not computed in the loaded descriptors data
        ValueError: the max number of directions to test is superior to the number of directions of the loaded descriptors data
    """
    previous_params, previous_nb_directions, computedDescriptors = load_computed_descriptors(filename = new_parameters.filename)

    # 0) Check that every unique descriptor to test is present in the loaded descriptors values
    lacking_descriptors = set(new_parameters.get_unique_descriptors()) - set(previous_params.get_unique_descriptors())
    if len(lacking_descriptors) > 0:
        raise ValueError(f"These descriptors cannot be tested (because they are not in the loaded descriptors file) : {lacking_descriptors}")
    
    # Check that the max number of directions (4 or 8) is inferior or equal to the number of directions of the loaded computed descriptors
    max_nb_directions = max(new_parameters.nb_directions)
    if max_nb_directions > previous_nb_directions:
        raise ValueError(f"Cannot test {max_nb_directions} directions, because the loaded descriptors are for {previous_nb_directions} directions")
    

    _show_parameters_LOAD_operation(previous_params, new_parameters, previous_nb_directions)
    
    # 1) Compute results (we go from 8 directions to 4 directions, because we rewrite the 'computedDescriptors' instance)
    global_results = SaveAndDisplayResults()

    for nb_directions in sorted(new_parameters.nb_directions, reverse=True):
        if nb_directions == 4: # reduce data to 4 directions, from an instance that computed data on 8 directions
            computedDescriptors = computedDescriptors.get_instance_from_8_directions_reduced_to_4_directions()

        # 4) Compute the results
        results_computing = ComputeResults()
        results_computing.compute_results_from_descriptors_combinations_and_classifiers(
            nb_directions       = nb_directions,
            computedDescriptors = computedDescriptors,
            parameters          = new_parameters
        )

        # 5) Add the results to the global results
        global_results.insert_results(nb_directions, results_computing)

    # 6) Display the results
    global_results.display_results()
    global_results.write_results_in_file(previous_params.database)


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

def _show_parameters_LOAD_operation(old_parameters: Parameters, new_parameters: Parameters, old_nb_directions: int):
    """Show the parameters. For the LOAD operation.

    Args:
        old_parameters (Parameters): the parameters from the loaded file
        new_parameters (Parameters): the new parameters to test
        old_nb_directions (int): the nb of directions from the loaded file
    """
    print("Parameters to test :")
    #print(f"\t• Database : {parameters.database.value}")
    print("\t• Classifier(s) : {}".format( ", ".join([c.value for c in new_parameters.classifiers]) ))
    print("\t• Nb of directions : {}".format( ", ".join([str(nb_dir) for nb_dir in new_parameters.nb_directions]) ))

    # To print descriptors combinations properly
    get_str_descriptor = lambda descriptor : descriptor if type(descriptor) is str else descriptor.value
    get_str_combination = lambda list_descriptors : [get_str_descriptor(descriptor) for descriptor in list_descriptors]
    sub_str_list = ", ".join([str(get_str_combination(sub_list)) for sub_list in new_parameters.descriptors_layout])
    print("\t• Descriptors : {}".format( str(sub_str_list) ))
    print()

    print("== Operation : LOAD descriptors ==")
    print(f"• (loaded) Database : {old_parameters.database.value}")
    print(f"• (loaded) Nb of directions : {old_nb_directions}")
    unique_descriptors_str = [get_str_descriptor(descriptor) for descriptor in old_parameters.get_unique_descriptors()]
    print(f"• (loaded) Computed descriptors : {sorted(unique_descriptors_str)}\n")
        
