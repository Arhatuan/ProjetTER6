from .utils.Parameters import Parameters
from .descriptors.utils.DescriptorsParameters import DescriptorsParameters
from .dataComputing.ComputeDescriptorsFromDatabase import ComputeDescriptorsFromDatabase
from .dataComputing.ComputeResults import ComputeResults
from .dataComputing.SaveAndDisplayResults import SaveAndDisplayResults



def manage(parameters: Parameters):
    show_parameters(parameters)

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




def show_parameters(parameters: Parameters):
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

