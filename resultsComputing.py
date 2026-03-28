import model_learning_2 as ModelLearning2
from model_learning_2 import descriptors

import numpy as np
import time
import os
from pathlib import Path
import re

from sklearn.metrics import confusion_matrix
from numpy import ndarray

def compute_X_data_from_descriptors(descriptors_values: dict, descriptors_list: list):
    X_data = []

    # Get the list of force degrees
    pattern = r"^f(-?\d+(\.\d+)?)$"
    forces_degrees = []
    forces_descriptors = []
    for descriptor in descriptors_list:
        match = re.match(pattern, descriptor)
        if match:
            forces_degrees.append(float(match.group(1)))
            forces_descriptors.append(descriptor)

    for i in range(len(descriptors_values[descriptors.DIST1])):
        sum_descriptors = []
        for descriptor in descriptors_list:
            match descriptor:
                case descriptors.RLM:       sum_descriptors.extend( descriptors_values[descriptors.RLM][i] )
                #case descriptors.FORCE:     sum_descriptors.extend( descriptors_values[descriptors.FORCE][i] )
                case descriptors.DISTANCE:  sum_descriptors.extend( descriptors_values[descriptors.DIST1][i] + descriptors_values[descriptors.DIST2][i] )
                case descriptors.DIST_MAX:  sum_descriptors.extend( max( descriptors_values[descriptors.DIST1][i], descriptors_values[descriptors.DIST2][i] ) )
                case descriptors.ANGLES:    sum_descriptors.extend( descriptors_values[descriptors.ANGLES][i] )

            if descriptor in forces_descriptors:
                sum_descriptors.extend(descriptors_values[descriptor][i])
        
        X_data.append(sum_descriptors)

    return X_data


def compute_results_for_descriptors_combinations_and_classifiers(descriptors_values: dict[str, list], descriptorsLayout: list[list[str]], classifiers: list[str], Y_data: list, nb_directions: int):
    results = dict()
    matrices = dict()

    # Gets the directions we have labels for (either 4 cardinal directions, or 8)
    match nb_directions:
        case 4: labels = ModelLearning2.SIMPLESHAPES_CLASSES_4
        case 8: labels = ModelLearning2.SIMPLESHAPES_CLASSES_8
        case _: raise ValueError(f"Unsupported number of directions for labels : {nb_directions}")

    for descriptors_combination in descriptorsLayout:
        X_data = compute_X_data_from_descriptors(descriptors_values, descriptors_combination)
        results['+'.join(descriptors_combination)] = dict()
        matrices['+'.join(descriptors_combination)] = dict()

        for classifier in classifiers:
            clf = ModelLearning2.get_trained_model(X_data, Y_data, classifier)
            scores = ModelLearning2.get_scores(clf, X_data, Y_data, printScores=False, padding="\t")
            
            results['+'.join(descriptors_combination)][classifier] = scores

            Y_predictions = clf.predict(X_data)
            conf_matrix = confusion_matrix(Y_data, Y_predictions, labels=labels)
            matrices['+'.join(descriptors_combination)][classifier] = conf_matrix

    return results, matrices


def write_results_in_file(results: dict, databaseName: str, matrices: dict = None):

    strResults = get_markdown_results_table(results, databaseName)
    str_specific_results = get_markdown_specific_results(results, matrices)

    final_str = strResults + str_specific_results

    match databaseName:
        case "SimpleShapes1" : filename = "S1_results_"
        case "SimpleShapes2" : filename = "S2_results_"
        case _ : filename = "unkwownDB_results_"
    filename += time.strftime("%Y_%m_%d_%Hh%Mm%S")

    filename = os.path.join("results", filename) # to the 'results' folder
    full_filename = os.path.join(Path(__file__).parent, filename)

    if not os.path.isdir("results"):
        os.mkdir("results")
    with open(full_filename + ".md", "w", encoding="utf-8") as file:
        file.write(final_str)
    
    print(f"Created results file at '{filename}'.")
    
def get_markdown_results_table(results: dict, databaseName: str):
    nb_directions_values = list(results.keys())
    descriptorsCombinations_values = list(results[nb_directions_values[0]].keys())
    classifiers_values = list(results[nb_directions_values[0]][descriptorsCombinations_values[0]].keys())

    # markdown file
    # Line 1
    strResults = f"# Results for {databaseName}\n\n"
    strResults += "|   |   | "
    for descriptorCombination in descriptorsCombinations_values:
        strResults += f" {descriptorCombination} |"

    # Line 2
    strResults += "\n|"
    strResults += "---|"*(2 + len(descriptorsCombinations_values))

    # Other lines
    for nb_directions in nb_directions_values:
        firstLine_nbDir = True
        for classifier in classifiers_values:
            if firstLine_nbDir:
                strResults += f"\n| {nb_directions} directions | {classifier.title()} | "
                firstLine_nbDir = False
            else:
                strResults += f"\n|   | {classifier.title()} | "
            
            for descriptorsCombination in descriptorsCombinations_values:
                resultsDescriptorsCombination = results[nb_directions][descriptorsCombination][classifier]
                strResults += "{:.2f} ± {:.2f} | ".format(np.mean(resultsDescriptorsCombination), np.std(resultsDescriptorsCombination))

    return strResults

def get_markdown_one_confusion_matrix(matrix: ndarray, nb_directions: int):
    match nb_directions:
        case 4: labels = ModelLearning2.SIMPLESHAPES_CLASSES_4
        case 8: labels = ['Above', 'Under', 'Left', 'Right', 'Abv L', 'Abv R', 'Und L', 'Und R'] # shorter labels (for display only)
        case _: raise ValueError(f"Unsupported number of directions for labels : {nb_directions}")

    # Line 1 + header separator
    str_matrix = "|   |"
    for label in labels:
        str_matrix += f" {label} |"
    str_matrix += "   |"
    str_matrix += "\n|" + "---|"*(len(labels) + 2)

    # Other lines
    for i, label in enumerate(labels):
        str_matrix += f"\n| {label} | "
        for j in range(len(labels)):
            str_matrix += f" {matrix[i,j]} |"
        str_matrix += f" {np.sum(matrix[i,:])} |" # line sum

    # Last line : column sums
    str_matrix += "\n|   |"
    for i in range(len(labels)):
        str_matrix += f" {np.sum(matrix[:,i])} |"
    str_matrix += f" {np.sum(matrix)} |" # total sum

    return str_matrix


def get_markdown_specific_results(results: dict, matrices: dict):
    nb_directions_values = list(results.keys())
    descriptorsCombinations_values = list(results[nb_directions_values[0]].keys())
    classifiers_values = list(results[nb_directions_values[0]][descriptorsCombinations_values[0]].keys())

    constructed_str = ""

    for nb_directions in nb_directions_values:
        for descriptorsCombination in descriptorsCombinations_values:
            for classifier in classifiers_values:
                constructed_str += "\n\n"
                constructed_str += f"## {descriptorsCombination} ({nb_directions} directions / {classifier.title()})"
                
                resultsDescriptorsCombination = results[nb_directions][descriptorsCombination][classifier]
                constructed_str += "\n\n| Final result | {:.2f} ± {:.2f} | ".format(np.mean(resultsDescriptorsCombination), np.std(resultsDescriptorsCombination))
                constructed_str += "\n|---|---|"
                constructed_str += "\n| Cross-validation results | {} |".format(" / ".join([str(round(crossResult, 4)) for crossResult in resultsDescriptorsCombination]))

                matrix = matrices[nb_directions][descriptorsCombination][classifier]
                matrix_md = get_markdown_one_confusion_matrix(matrix, nb_directions)
                constructed_str += "\n\n" + matrix_md

    return constructed_str

