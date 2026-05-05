import json
import model_learning_2 as ModelLearning2
import resultsComputing as ResultsComputing
from extendedRLM.model_learning import *

import argparse
import os
from pathlib import Path
import time
import numpy as np
import re

PATH_DB_SIMPLESHAPES1 = os.path.join(Path(__file__).parent, "images", "SimpleShapes1")
PATH_DB_SIMPLESHAPES2 = os.path.join(Path(__file__).parent, "images", "SimpleShapes2")

PATH_ANNOTATIONS_SIMPLESHAPES1 = os.path.join(Path(__file__).parent, "annotations", "SimpleShapes1.csv")
PATH_ANNOTATIONS_SIMPLESHAPES2 = os.path.join(Path(__file__).parent, "annotations", "SimpleShapes2.csv")


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-c", "--classifiers",
                        choices=["MLP", "RF"],
                        nargs="*",
                        default=["MLP"],
                        type=str.upper,
                        help="Classifier to use. Choices are : MLP (multi-layer perceptron) and RF (Random forests). Default = MLP")
    parser.add_argument("--db", "--database",
                        choices=["S1", "S2"],
                        default="S1",
                        type=str.upper,
                        help="Database to use. Choices are : S1 (SimpleShapes1) and S2 (SimpleShapes2). Default = S1")
    
    parser.add_argument("-d", "--descriptors",
                        nargs="*",
                        type=lambda string: str.split(string, '+'),
                        default=[["RLM", "F2"]],
                        metavar="{RLM,F0,F2,DIST,ANGLE}",
                        help="Characteristics (or descriptors) to use. You can give multiple combinations. (f. ex. 'rlm+force rlm+dist+angle')")
    
    parser.add_argument("-f", "--force",
                        type=float,
                        default=2,
                        help="Degree of force for the Force descriptor. Default = 2")
    parser.add_argument("--nb_dir",
                        nargs="*",
                        type=int,
                        choices=[4,8],
                        default=[4],
                        help="The number of directions to evaluate. Either 4 or 8. Choose the 2 for computing results for both of them separately. Default = 4 only")


    args = parser.parse_args()

    # Decide the path to the image database + get the annotations
    match args.db:
        case "S1":
            path_db = PATH_DB_SIMPLESHAPES1
            annotations = load_annotations(PATH_ANNOTATIONS_SIMPLESHAPES1)
        case "S2":
            path_db = PATH_DB_SIMPLESHAPES2
            annotations = load_annotations(PATH_ANNOTATIONS_SIMPLESHAPES2)
        case _:
            parser.error("The database argument isn't correct.")

    # The classifier to train a model on
    classifiers = []
    for classifier in args.classifiers:
        match classifier:
            case "MLP": classifiers.append(ModelLearning2.classifiers.MLP)
            case "RF":  classifiers.append(ModelLearning2.classifiers.RF)
            case _: parser.error("Unsupported classifier")

    # The list of descriptors's combinations
    descriptorsLayout = []
        # check that all the descriptors exist
    for listDescriptors in args.descriptors:
        new_list_descriptors = []
        for descriptor in listDescriptors:
            match str.upper(descriptor):
                case "RLM": new_list_descriptors.append(ModelLearning2.descriptors.RLM)
                case "FORCE": new_list_descriptors.append(ModelLearning2.descriptors.FORCE)
                case "DIST": new_list_descriptors.append(ModelLearning2.descriptors.DISTANCE)
                case "ANGLE": new_list_descriptors.append(ModelLearning2.descriptors.ANGLES)
                case _ if (m := re.match(r"^f(-?\d+(\.\d+)?)$", str.lower(descriptor))):
                    new_list_descriptors.append(str.lower(descriptor))
                case _: parser.error(f"Unknown descriptor '{descriptor}' in the list {listDescriptors}")
        descriptorsLayout.append(new_list_descriptors)
    
    return (path_db, annotations, classifiers, descriptorsLayout, args.force, args.nb_dir)


def show_parameters(path_db: str, classifiers: list[str], descriptorsLayout: list[str], force: float, nb_directions_list: list[int]):
    
    print("Parameters :")
    print(f"\t• Database : {path_db.split(os.path.sep)[-1]}")
    print("\t• Classifier : {}".format( ", ".join([c.capitalize() for c in classifiers]) ) )
    #if "force" in [descriptor.lower() for descriptor in descriptors]: print(f"\t• Force : {force}")
    print(f"\t• Nb of directions : {", ".join([str(nb_directions) for nb_directions in nb_directions_list])}")
    print(f"\t• Descriptors : {", ".join([str(listDescriptors) for listDescriptors in descriptorsLayout])}")
    print()


if __name__ == '__main__':

    # 1) Gets the parameters from the command line, and show them for confirmation
    (path_db, annotations, classifiers, descriptorsLayout, force, nb_directions_list) = parse_args()
    show_parameters(path_db, classifiers, descriptorsLayout, force, nb_directions_list)

    unique_descriptors_to_compute = list(set([descriptor for descriptors_list in descriptorsLayout for descriptor in descriptors_list]))
    global_results = dict()
    global_matrices = dict()

    # 2) Compute the descriptors values, the annotations, and the results for each nb_directions / descriptors_combination / classifier
    for nb_directions in nb_directions_list:
        global_results[nb_directions] = dict()

        # 2) Compute the descriptors values (and the annotations at the same time)
        timeStart = time.time()
        print(f"-> (computing descriptors) For {nb_directions} directions :")
        descriptors_values, Y1 = ModelLearning2.compute_descriptors_and_Y_data_on_SimpleShape(path_db, annotations, (68, 1, 84, 255), 3, force, unique_descriptors_to_compute, nb_directions)
        print("\tTime taken : {:.1f}s".format(time.time() - timeStart))

        results_descriptors_per_classifier, matrices_descriptors_per_classifier = ResultsComputing.compute_results_for_descriptors_combinations_and_classifiers(descriptors_values, descriptorsLayout, classifiers, Y1, nb_directions)

        global_results[nb_directions] = results_descriptors_per_classifier
        global_matrices[nb_directions] = matrices_descriptors_per_classifier


    # 4) Show the results
    for nb_directions in nb_directions_list:
        print(f"\n• Results for {nb_directions} :")
        
        for descriptors_combination in global_results[nb_directions].keys():
            print("\n\t• Results for {} :".format(descriptors_combination))

            for classifier in classifiers:
                scores = global_results[nb_directions][descriptors_combination][classifier]
                strClassifier = "{} : ".format(classifier.capitalize())
                print("\t\t- {:<25}\t {:.2f} ± {:.2f}".format(strClassifier, scores.mean(), scores.std()))

    ResultsComputing.write_results_in_file(global_results, path_db.split(os.path.sep)[-1], global_matrices)

    # with open("annotations/SpatialSense", 'r') as f:
    #     SpatialSense = json.load(f)