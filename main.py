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
    
    parser.add_argument("-c", "--classifier",
                        choices=["MLP", "RF"],
                        default="MLP",
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
                        default=["RLM", "Force"],
                        help="Characteristics (or descriptors) to use. Choices are : RLM, Force and Dist. Default = [RLM, Force]")
    parser.add_argument("-f", "--force",
                        type=float,
                        default=2,
                        help="Degree of force for the Force descriptor. Default = 2")


    args = parser.parse_args()

    # Decide the path to the image database + get the annotations
    match args.db:
        case "S1":
            path_db = PATH_DB_SIMPLESHAPES1
            annotations = load_annotations(PATH_ANNOTATIONS_SIMPLESHAPES1)
        case "S2":
            path_db = PATH_DB_SIMPLESHAPES2
            annotations = load_annotations(PATH_ANNOTATIONS_SIMPLESHAPES1)
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
    match args.classifier:
        case "MLP": classifier = ModelLearning2.classifiers.MLP
        case "RF":  classifier = ModelLearning2.classifiers.RF
        case _: parser.error("Unsupported classifier")

    
    return (path_db, annotations, classifier, descriptors, args.force)



if __name__ == '__main__':

    (path_db, annotations, classifier, descriptors, force) = parse_args()

    # with open("annotations/SpatialSense", 'r') as f:
    #     SpatialSense = json.load(f)

    print(f"Database : {path_db.split(os.path.sep)[-1]}")
    print(f"Classifier : {classifier.capitalize()}")
    print(f"Descriptors : {", ".join(descriptors)}")
    if "force" in [descriptor.lower() for descriptor in descriptors]: print(f"Force : {force}")
    print()
    time.sleep(2)
    
    X1, Y1 = ModelLearning2.compute_extendedRLM_on_SimpleShape_v2(path_db, annotations, (68, 1, 84, 255), 3, force, descriptors)
    
    #X2, Y2 = ModelLearning2.compute_extendedRLM_on_SimpleShape("images/SimpleShapes2", SimpleShape2, (68, 1, 84, 255), 3, 2)

    print(f"Results for database '{path_db.split(os.path.sep)[-1]}'")
    print(f"\t-> descriptors : {", ".join(descriptors)}")
    if "force" in [descriptor.lower() for descriptor in descriptors]: print(f"\t-> force : {force}")
    print(f"\t-> classifier : {classifier.capitalize()}")
    print()

    ModelLearning2.train_model_v2(X1, Y1, True, classifier)