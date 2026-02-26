import image_2 as Image2
import types

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

descriptors = types.SimpleNamespace()
descriptors.RLM = "rlm"
descriptors.FORCE = "force"
descriptors.DISTANCE = "distance"

classifiers = types.SimpleNamespace()
classifiers.MLP = "multi-layer perceptron"
classifiers.RF = "random forest"


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

    return clf

def compute_extendedRLM_on_SimpleShape_v2(folder, annotations, background, step, force, descriptors_list):
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
    for row in annotations:
        print(f"img-{row['obj1']}-{row['obj2']}-{row['nb']}.png")

        if row['nb'] != "?" and row['diff'] != '4':
            img_name = f"{folder}/img-{row['obj1']}-{row['obj2']}-{row['nb']}.png"
            rlm1, rlm2, forces, dist1, dist2 = Image2.image_processing_v2(img_name, background, step, force)

            # Concatenate the descriptors' vectors 
            #   (for example : 1000 data with each 5 descriptors' vectors of length 120 -> total vector of length 1000 * 120 * 5 = 600 000)
            sumDescriptors = []
            for descriptor in descriptors_list:
                match descriptor:
                    case descriptors.RLM:       sumDescriptors = sumDescriptors + (rlm1 + rlm2)
                    case descriptors.FORCE:     sumDescriptors = sumDescriptors + (forces)
                    case descriptors.DISTANCE:  sumDescriptors = sumDescriptors + (dist1 + dist2)

            X_data.append(sumDescriptors)
            Y_data.append(row["rel"])
    return X_data, Y_data