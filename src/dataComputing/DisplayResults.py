from .ComputeResults import ComputeResults

class DisplayResults:
    """Keep results for 4 directions and 8 directions, and display them in the terminal"""

    results_4_directions: ComputeResults
    results_8_directions: ComputeResults

    def __init__(self):
        self.results_4_directions = None
        self.results_8_directions = None


    def insert_results(self, nb_directions: int, computed_results: ComputeResults):
        """Insert the results belonging to the given number of directions.

        Args:
            nb_directions (int): the number of directions (4 or 8) the data belong to.
            computed_results (ComputeResults): the results to insert.

        Raises:
            ValueError: unsupported number of directions (must be 4 or 8).
        """
        match nb_directions:
            case 4: self.results_4_directions = computed_results
            case 8: self.results_8_directions = computed_results
            case _: raise ValueError(f"Unsupported number of directions : {nb_directions}")

    def __show_result_for_nb_directions(self, nb_directions: int):
        """Display results for data belonging to the given number of directions (4 or 8)"""
        match nb_directions:
            case 4: results = self.results_4_directions
            case 8: results = self.results_8_directions
            case _: raise ValueError(f"Unsupported number of directions : {nb_directions}")

        print(f"\n• Results for {nb_directions} directions :")

        if results is None:
            print("\t\tNone") # nothing to show
            return
        
        for descriptors_combination in results.scores.keys():
            print(f"\n\t• Results for {descriptors_combination}")

            for classifier in results.scores[descriptors_combination].keys():
                # scores = 5 scores for each subset in cross-validation
                scores = results.scores[descriptors_combination][classifier]
                strClassifier = "{} : ".format(classifier.value.capitalize())
                print("\t\t- {:<25}\t {:.2f} ± {:.2f}".format(strClassifier, scores.mean(), scores.std()))
    
    def display_results(self):
        """Display the results for each direction, then each descriptors's combination, then each classifier
        """
        self.__show_result_for_nb_directions(4)
        self.__show_result_for_nb_directions(8)
    

