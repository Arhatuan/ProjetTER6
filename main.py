import json
import model_learning_2 as ModelLearning2
from extendedRLM.model_learning import *

import argparse
import os
from pathlib import Path
import time

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
                        choices=["RLM", "FORCE", "DIST"],
                        nargs="*",
                        type=str.upper,
                        default=["RLM", "FORCE"],
                        help="Characteristics (or descriptors) to use. Choices are : RLM, Force and Dist. Default = [RLM, Force]")
    parser.add_argument("-f", "--force",
                        type=float,
                        default=2,
                        help="Degree of force for the Force descriptor. Default = 2")
    parser.add_argument("--nb_dir",
                        type=int,
                        choices=[4,8],
                        default=4,
                        help="The number of directions to evaluate. Either 4 or 8. Default = 2")


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

    # Make the list of descriptors to be used for training the model
    descriptors = []
    for descriptor in args.descriptors:
        match descriptor:
            case "RLM": descriptors.append(ModelLearning2.descriptors.RLM)
            case "FORCE": descriptors.append(ModelLearning2.descriptors.FORCE)
            case "DIST": descriptors.append(ModelLearning2.descriptors.DISTANCE)

    # The classifier to train a model on
    classifiers = []
    for classifier in args.classifiers:
        match classifier:
            case "MLP": classifiers.append(ModelLearning2.classifiers.MLP)
            case "RF":  classifiers.append(ModelLearning2.classifiers.RF)
            case _: parser.error("Unsupported classifier")

    
    return (path_db, annotations, classifiers, descriptors, args.force, args.nb_dir)


def show_parameters(path_db: str, classifiers: list[str], descriptors: list[str], force: float, nb_directions: int):
    print("Parameters :")
    print(f"\t• Database : {path_db.split(os.path.sep)[-1]}")
    print("\t• Classifier : {}".format( ", ".join([c.capitalize() for c in classifiers]) ) )
    print(f"\t• Descriptors : {", ".join(descriptors)}")
    if "force" in [descriptor.lower() for descriptor in descriptors]: print(f"\t• Force : {force}")
    print(f"\t• Nb of directions : {nb_directions}")
    print()


if __name__ == '__main__':

    # 1) Gets the parameters from the command line, and show them for confirmation
    (path_db, annotations, classifiers, descriptors, force, nb_directions) = parse_args()
    show_parameters(path_db, classifiers, descriptors, force, nb_directions)
    
    # 2) Compute the descriptors values (and the annotations at the same time)
    timeStart = time.time()
    X1, Y1 = ModelLearning2.compute_extendedRLM_on_SimpleShape_v2(path_db, annotations, (68, 1, 84, 255), 3, force, descriptors, nb_directions)
    print("Time taken for computing descriptors : {:.1f}s".format(time.time() - timeStart))

    # 3) Train a classifier on the data, and show its scores
    for classifier in classifiers:
        print("\n• Results for '{}' :".format(classifier.capitalize()))
        clf = ModelLearning2.get_trained_model(X1, Y1, classifier)
        ModelLearning2.print_scores(clf, X1, Y1, "\t")
        ModelLearning2.print_confusion_matrix(clf, X1, Y1, nb_directions, "\t")
    
    # with open("annotations/SpatialSense", 'r') as f:
    #     SpatialSense = json.load(f)