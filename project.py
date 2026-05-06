import argparse
import re

from src.utils.enumerations import Database, Classifier, ReferencePointMode
from src.descriptors.utils.DescriptorEnum import Descriptor, FORCE_REGEX
from src.utils.Parameters import Parameters
from src import manager

def parse_args() -> Parameters:
    """Parse arguments to decide the parameters of the program (classifiers, database, descriptors, number of directions...)

    Returns:
        Parameters: the parameters enclosed in a `Parameters` instance
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--classifiers",
                        choices=["MLP", "RF"],
                        nargs="*",
                        default=["MLP"],
                        type=str.upper,
                        help="Classifier to use. Choices are : MLP (Multi-layer Perceptron), RF (Random Forests). Default = MLP")
    
    parser.add_argument("--db", "--database",
                        choices=["S1", "S2", "SIG", "SH"],
                        default="S1",
                        type=str.upper,
                        help="Database to use. Choices are : S1 (SimpleShapes1), S2 (SimpleShapes2), SIG, SH (SharvitSR). Default = S1")
    
    parser.add_argument("-d", "--descriptors",
                        nargs="*",
                        type=lambda string: str.split(string, '+'),
                        default=[["RLM", "F2"]],
                        metavar="{RLM,F0,F2,DIST,ANGLE,ANGLE8}",
                        help="Descriptors to use. You can give multiple combinations (for ex. 'rlm+f2 rlm+dist+angle')")
    
    parser.add_argument("-n", "--nb_dir",
                        nargs="*",
                        type=int,
                        choices=[4,8],
                        default=[4],
                        help="The number of directions to evaluate. Either 4 or 8. Choose the 2 of them for computing results for both of them separately. Default = 4 only")

    parser.add_argument("--rp-mode",
                        choices=["DETERMINISTIC", "RANDOM_FULL", "RANDOM_BORDER", "RANDOM_CENTER"],
                        default="DETERMINISTIC",
                        type=str.upper,
                        help="How to choose Rp (reference point): deterministic or random modes.")
    
    parser.add_argument("--rp-border-size",
                        type=int,
                        default=20,
                        help="Border size in pixels for RANDOM_BORDER and RANDOM_CENTER Rp modes. Default = 20")
    
    parser.add_argument("--iterations",
                        type=int,
                        default=1,
                        help="Number of independent training iterations. Default = 1")
    
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Base seed used to generate independent iterations. Default = 42")
    
    args = parser.parse_args()

    parameters = Parameters()

    # Decide the database
    match args.db:
        case "S1": parameters.set_database(Database.S1)
        case "S2": parameters.set_database(Database.S2)
        case "SIG": parameters.set_database(Database.SIG)
        case "SH": parameters.set_database(Database.SHARVITSR)
        case _: parser.error(f"Unsupported database : '{args.db}'")

    # Decide the classifier(s) to train a model on
    for classifier in args.classifiers:
        match classifier:
            case "MLP": parameters.add_classifier(Classifier.MLP)
            case "RF": parameters.add_classifier(Classifier.RF)
            case _: parser.error(f"Unsupported classifier : {classifier}")

    # Decide the descriptor's combinations
    for listDescriptors in args.descriptors:
        list_combination_descriptors = []
        for descriptor in listDescriptors:
            match str.upper(descriptor):
                case "RLM": list_combination_descriptors.append(Descriptor.RLM)
                case "DIST": list_combination_descriptors.append(Descriptor.DISTANCE)
                case "ANGLE": list_combination_descriptors.append(Descriptor.ANGLES)
                case "ANGLE8": list_combination_descriptors.append(Descriptor.ANGLES8)
                case _ if (m := re.match(FORCE_REGEX, str.lower(descriptor))):
                    list_combination_descriptors.append(str.lower(descriptor))
                case _: parser.error(f"Unknown descriptor '{descriptor}' in the list {listDescriptors}")
        parameters.add_descriptors_combination(list_combination_descriptors)

    # Decide the number of directions
    for nb_dir in args.nb_dir:
        match nb_dir:
            case 4: parameters.add_nb_directions(4)
            case 8: parameters.add_nb_directions(8)
            case _: parser.error(f"Unsupported number of directions : {nb_dir}")

    # Decide how Rp is selected
    match args.rp_mode:
        case "DETERMINISTIC": parameters.set_rp_mode(ReferencePointMode.DETERMINISTIC)
        case "RANDOM_FULL": parameters.set_rp_mode(ReferencePointMode.RANDOM_FULL)
        case "RANDOM_BORDER": parameters.set_rp_mode(ReferencePointMode.RANDOM_BORDER)
        case "RANDOM_CENTER": parameters.set_rp_mode(ReferencePointMode.RANDOM_CENTER)
        case _: parser.error(f"Unsupported rp mode: {args.rp_mode}")

    if args.rp_border_size < 0:
        parser.error("rp-border-size must be >= 0")
    parameters.set_rp_border_size(args.rp_border_size)

    if args.iterations < 1:
        parser.error("iterations must be >= 1")
    parameters.set_nb_iterations(args.iterations)
    parameters.set_random_seed(args.seed)

    return parameters



if __name__ == "__main__":
    # Get the arguments from the command line
    parameters = parse_args()
    # Launch the program with the arguments as parameters
    manager.manage(parameters)
