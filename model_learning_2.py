import image_2 as Image2
import types
import numpy as np
import re

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix

descriptors = types.SimpleNamespace()
descriptors.RLM = "rlm"
descriptors.FORCE = "force"
descriptors.DISTANCE = "distance"
descriptors.DIST_MAX = "distance_max"
descriptors.DIST1 = "dist1"
descriptors.DIST2 = "dist2"
descriptors.ANGLES = "angles"

classifiers = types.SimpleNamespace()
classifiers.MLP = "multi-layer perceptron"
classifiers.RF = "random forest"

SIMPLESHAPES_CLASSES_8 = ['Above', 'Under', 'Left', 'Right', 'Above Left', 'Above Right', 'Under Left', 'Under Right']
SIMPLESHAPES_CLASSES_4 = ['Above', 'Under', 'Left', 'Right']


def train_model_v2(X=None, Y=None, print_scores=False, classifier=classifiers.MLP):
    """
    Train a MLP Classifier from X data on Y classes.
    :param X: Data to learn from ; usually RLM values of both objects and the
    :param Y: Classes where the data of X[i] corresponds to the class Y[i].
    :param print_scores: boolean, used to allow or not the printing of accuracy found during training.
    :return: the trained model.
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2) # 0.2

    # Choose the classifier
    match classifier:
        case classifiers.MLP:   clf = MLPClassifier(hidden_layer_sizes=(4, 448), solver='adam', max_iter=1000)  # 224 
        case classifiers.RF:    clf = RandomForestClassifier()
        case _:                 raise ValueError(f"Unsupported classifier : {classifier}")

    clf.fit(X_train, Y_train)

    if print_scores:
        scores = cross_val_score(clf, X, Y, cv=5)
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
        print("Results per sub-base in cross-validation : \n", scores)
        
        Y_predictions = clf.predict(X)
        conf_matrix = confusion_matrix(Y, Y_predictions, labels=SIMPLESHAPES_CLASSES_8)
        print("\nConfusion matrix :\n", conf_matrix)


    return clf

def compute_extendedRLM_on_SimpleShape_v2(folder, annotations, background, step, force, descriptors_list, nb_directions):
    """
    Compute the histograms (RLMs and forces) of all images of the SimpleShape dataset (either S1 or S2).

    :param folder: folder where images are stocked.
    :param annotations: path of the file containing the annotations of a SimpleShape dataset.
    :param background: Background color of the images.
    :param step: Step of the angle to use for the computation of the histograms.
    :param force: type of force to use (0, or 2) for the computation of the f-histogram.
    :return: Two lists: (Histograms and classes).
    """
    X_data = []
    Y_data = []
    total_length = len(annotations)
    i = 0

    # Gets the directions we're testing (either 4 cardinal directions, or 8)
    match nb_directions:
        case 4: tested_directions = SIMPLESHAPES_CLASSES_4
        case 8: tested_directions = SIMPLESHAPES_CLASSES_8
        case _: raise ValueError(f"Unsupported number of directions to test : {nb_directions}")

    for row in annotations:
        
        # Print the current state of computing the descriptors for each image in the database
        i += 1
        strImgProcessed = f"img-{row['obj1']}-{row['obj2']}-{row['nb']}.png"
        strProgress = "(progress: {:2.1%})".format(i/total_length)
        print("{:<16} {:>18}".format(strImgProcessed, strProgress), end="\r")
        if i == total_length: print() # to cancel the last carriage return character '\r'

        if (row['nb'] != "?" 
            # and row['diff'] != '4' # exclude max difficulty
            and row['rel'] in tested_directions): # number of directions tested

            img_name = f"{folder}/img-{row['obj1']}-{row['obj2']}-{row['nb']}.png"
            rlm1, rlm2, forces, dist1, dist2, angles = Image2.image_processing_v2(img_name, background, step, force)

            # Concatenate the descriptors' vectors 
            #   (for example : 1000 data with each 5 descriptors' vectors of length 120 -> total vector of length 1000 * 120 * 5 = 600 000)
            sumDescriptors = []
            for descriptor in descriptors_list:
                match descriptor:
                    case descriptors.RLM:       sumDescriptors = sumDescriptors + (rlm1 + rlm2)
                    case descriptors.FORCE:     sumDescriptors = sumDescriptors + (forces)
                    case descriptors.DISTANCE:  sumDescriptors = sumDescriptors + (dist1 + dist2)
                    case descriptors.ANGLES:    sumDescriptors = sumDescriptors + (angles)

            X_data.append(sumDescriptors)
            Y_data.append(row["rel"])
    
    return X_data, Y_data


def print_current_state_computing_descriptors(nb: int, total_length: int, img_name: str):
    strProgress = "(progress: {:2.1%})".format(nb/total_length)
    print("{:<16} {:>18}".format(img_name, strProgress), end="\r")
    if nb == total_length: print() # to cancel the last carriage return character '\r'

def compute_descriptors_and_Y_data_on_SimpleShape(folder, annotations, background, step, force, descriptors_list, nb_directions):
    descriptorsValues = { k:[] for k in set(descriptors.__dict__.values()) | set(descriptors_list)}
    Y_data = []
    total_length = len(annotations)

    # Gets the directions we're testing (either 4 cardinal directions, or 8)
    match nb_directions:
        case 4: tested_directions = SIMPLESHAPES_CLASSES_4
        case 8: tested_directions = SIMPLESHAPES_CLASSES_8
        case _: raise ValueError(f"Unsupported number of directions to test : {nb_directions}")

    # Get the list of force degrees
    pattern = r"^f(-?\d+(\.\d+)?)$"
    forces_degrees = []
    forces_descriptors = []
    for descriptor in descriptors_list:
        match = re.match(pattern, descriptor)
        if match:
            forces_degrees.append(float(match.group(1)))
            forces_descriptors.append(descriptor)

    for i, row in enumerate(annotations):
    
        # Print the current state of computing the descriptors for each image in the database
        strImgProcessed = f"img-{row['obj1']}-{row['obj2']}-{row['nb']}.png"
        print_current_state_computing_descriptors(i+1, total_length, strImgProcessed)

        if (row['nb'] != "?" 
            # and row['diff'] != '4' # exclude max difficulty
            and row['rel'] in tested_directions): # number of directions tested

            img_name = f"{folder}/img-{row['obj1']}-{row['obj2']}-{row['nb']}.png"
            rlm1, rlm2, forces, dist1, dist2, angles = Image2.image_processing_v3(
                                                            img_name, background, step, 
                                                            list_forces=    forces_degrees,
                                                            computeRLM=     descriptors.RLM in descriptors_list,
                                                            computeDist=    descriptors.DISTANCE in descriptors_list or descriptors.DIST_MAX in descriptors_list,
                                                            computeAngles=  descriptors.ANGLES in descriptors_list)
            

            descriptorsValues[descriptors.RLM].append( (rlm1 + rlm2) )
            #descriptorsValues[descriptors.FORCE].append( forces )
            descriptorsValues[descriptors.DIST1].append( dist1 )
            descriptorsValues[descriptors.DIST2].append( dist2 )
            descriptorsValues[descriptors.ANGLES].append( angles )

            for force_descriptor, force_degree in zip(forces_descriptors, forces_degrees):
                descriptorsValues[force_descriptor].append( forces[force_degree] )

            Y_data.append(row["rel"])
    
    return descriptorsValues, Y_data

def compute_X_data_from_descriptors(descriptors_values: dict, descriptors_list: list):
    X_data = []

    for i in range(len(descriptors_values[descriptors.DIST1])):
        sum_descriptors = []
        for descriptor in descriptors_list:
            match descriptor:
                case descriptors.RLM:       sum_descriptors.extend( descriptors_values[descriptors.RLM][i] )
                case descriptors.FORCE:     sum_descriptors.extend( descriptors_values[descriptors.FORCE][i] )
                case descriptors.DISTANCE:  sum_descriptors.extend( descriptors_values[descriptors.DIST1][i] + descriptors_values[descriptors.DIST2][i] )
                case descriptors.DIST_MAX:  sum_descriptors.extend( max( descriptors_values[descriptors.DIST1][i], descriptors_values[descriptors.DIST2][i] ) )
                case descriptors.ANGLES:    sum_descriptors.extend( descriptors_values[descriptors.ANGLES][i] )
            
        
        X_data.append(sum_descriptors)

    return X_data

def get_trained_model(X: list, Y: list, classifier=classifiers.MLP):
    """Train a model given data (X) and its ground truth (Y)

    Args:
        X (list): data to learn from. They are values obtained by descriptors.
        Y (list): the ground truth of the X data. (Y[i] corresponds to the data X[i])
        classifier (str, optional): the classifier to train. Defaults to classifiers.MLP.

    Returns:
        classifier: the trained classifier
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2) # 0.2

    # Choose the classifier
    match classifier:
        case classifiers.MLP:   clf = MLPClassifier(hidden_layer_sizes=(4, 448), solver='adam', max_iter=1000)  # 224 
        case classifiers.RF:    clf = RandomForestClassifier()
        case _:                 raise ValueError(f"Unsupported classifier : {classifier}")

    clf.fit(X_train, Y_train)
    return clf

def get_scores(clf, X: list, Y: list, printScores = False, padding = ""):
    """Print the scores of the classifier on the given data.
    It makes a cross-validation on 5 subsets of the data.

    Args:
        clf: the classifier to test
        X (list): data to test. They are values obtained by descriptors. 
        Y (list): the ground truth of the X data. (Y[i] corresponds to the data X[i])
    """
    scores = cross_val_score(clf, X, Y, cv=5)
    if printScores:
        print(padding + "%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
        #print("Results per sub-base in cross-validation : \n", scores)
    return scores


def print_confusion_matrix(clf, X: list, Y: list, nb_directions: int, padding = ""):
    # Gets the directions we have labels for (either 4 cardinal directions, or 8)
    match nb_directions:
        case 4: labels = SIMPLESHAPES_CLASSES_4
        case 8: labels = SIMPLESHAPES_CLASSES_8
        case _: raise ValueError(f"Unsupported number of directions for labels : {nb_directions}")
    
    Y_predictions = clf.predict(X)
    conf_matrix = confusion_matrix(Y, Y_predictions, labels=labels)

    if nb_directions == 8:
        labels = ['Above', 'Under', 'Left', 'Right', 'Abv L', 'Abv R', 'Und L', 'Und R'] # shorter labels (for display only)

    max_str_length = max([len(label) for label in labels] + [len(str(np.max(conf_matrix)))])

    # First line (header)
    constructed_string = padding + " "*(max_str_length+3)
    for label in labels:
        constructed_string += ("{:^"+str(max_str_length)+"}").format(label)
        constructed_string += " | "
    constructed_string = constructed_string[:-2]

    # Other lines (matrix values)
    for i, label in enumerate(labels):
        constructed_string += "\n" + padding
        constructed_string += ("{:<"+str(max_str_length)+"}").format(label)
        constructed_string += " "*2
        for j in range(len(conf_matrix[i])):
            constructed_string += ("{:^"+str(max_str_length+3)+"}").format(conf_matrix[i,j])
        constructed_string += ("{:^"+str(max_str_length+3)+"}").format(np.sum(conf_matrix[i,:]))

    # Last line (column sum)
    constructed_string += "\n" + padding
    constructed_string += " "*(max_str_length+2)
    for i, label in enumerate(labels):
        constructed_string += ("{:^"+str(max_str_length+3)+"}").format(np.sum(conf_matrix[:,i]))
    constructed_string += ("{:^"+str(max_str_length+3)+"}").format(np.sum(conf_matrix))

    print("\n"+ padding + "Confusion matrix :\n", constructed_string)
    

