import model_learning_2 as ModelLearning2
from model_learning_2 import descriptors

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


def compute_results_for_descriptors_combinations_and_classifiers(descriptors_values: dict[str, list], descriptorsLayout: list[list[str]], classifiers: list[str], Y_data: list):
    results = dict()

    for descriptors_combination in descriptorsLayout:
        X_data = ModelLearning2.compute_X_data_from_descriptors(descriptors_values, descriptors_combination)
        results['+'.join(descriptors_combination)] = dict()

        for classifier in classifiers:
            clf = ModelLearning2.get_trained_model(X_data, Y_data, classifier)
            scores = ModelLearning2.get_scores(clf, X_data, Y_data, printScores=False, padding="\t")
            
            results['+'.join(descriptors_combination)][classifier] = scores

    return results