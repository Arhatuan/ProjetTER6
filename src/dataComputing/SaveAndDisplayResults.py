import numpy as np
from numpy import ndarray
import time
import os
from pathlib import Path

from ..utils.enumerations import Database
from .DisplayResults import DisplayResults

PATH_RESULTS_DIR = os.path.join(Path(__file__).parent.parent.parent, "results")

class SaveAndDisplayResults(DisplayResults):
    """Keep results for 4 directions and 8 directions, then save them into a file, and/or display them in the terminal"""

    def write_results_in_file(self, database: Database):
        # 1) Get the string text to save in a file
        str_results = self.__get_markdown_results_table(database)
        str_specific_results = self.__get_markdown_specific_results()
        final_str = str_results + str_specific_results

        # 2) Choose the file's name
        match database:
            case Database.S1: filename = "S1_results_"
            case Database.S2: filename = "S2_results_"
            case _: filename = "unknownDB_results_"
        filename += time.strftime("%Y_%m_%d_%Hh%Mm%S")

        # 3) Get the absolute path to where we're saving the file
        full_filename = os.path.join(PATH_RESULTS_DIR, filename)

        if not os.path.isdir(PATH_RESULTS_DIR):
            os.mkdir(PATH_RESULTS_DIR)
        with open(full_filename + ".md", "w", encoding="utf-8") as file:
            file.write(final_str)

        print(f"Created results file at 'results/{filename}'")


    def __get_markdown_results_table(self, database: Database) -> str:
            assert self.results_4_directions is not None or self.results_8_directions is not None, "There must be results for 4 or 8 directions before saving them"
            
            results = self.results_4_directions if self.results_4_directions is not None else self.results_8_directions
            descriptorsCombinations_values = list(results.scores.keys())
            assert len(descriptorsCombinations_values) > 0, "There must be at least one descriptor's combination"
            classifiers_values = list(results.scores[descriptorsCombinations_values[0]].keys())
            
            # Line 1
            strResults = f"# Results for {database.value}\n\n"
            strResults += "|   |   | "
            for descriptorCombination in descriptorsCombinations_values:
                strResults += f" {descriptorCombination} |"

            # Line 2
            strResults += "\n|"
            strResults += "---|"*(2 + len(descriptorsCombinations_values))

            # Other lines
            for results, nb_directions in zip([self.results_4_directions, self.results_8_directions], [4,8]):
                if results is None:
                    continue

                firstLine_nbDir = True
                for classifier in classifiers_values:
                    if firstLine_nbDir:
                        strResults += f"\n| {nb_directions} directions | {classifier.value.title()} | "
                        firstLine_nbDir = False
                    else:
                        strResults += f"\n|   | {classifier.value.title()} | "
                    
                    for descriptorsCombination in descriptorsCombinations_values:
                        resultsDescriptorsCombination = results.scores[descriptorsCombination][classifier]
                        strResults += "{:.2f} ± {:.2f} | ".format(np.mean(resultsDescriptorsCombination), np.std(resultsDescriptorsCombination))

            return strResults
    
    
    def __get_markdown_one_confusion_matrix(self, matrix: ndarray, nb_directions: int) -> str:
        match nb_directions:
            case 4: labels = ['Above', 'Under', 'Left', 'Right']
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
    


    def __get_markdown_specific_results(self) -> str:
        results = self.results_4_directions if self.results_4_directions is not None else self.results_8_directions
        descriptorsCombinations_values = list(results.scores.keys())
        assert len(descriptorsCombinations_values) > 0, "There must be at least one descriptor's combination"
        classifiers_values = list(results.scores[descriptorsCombinations_values[0]].keys())
        
        constructed_str = ""

        for results, nb_directions in zip([self.results_4_directions, self.results_8_directions], [4,8]):
            if results is None:
                continue
            
            for descriptorsCombination in descriptorsCombinations_values:
                for classifier in classifiers_values:
                    constructed_str += "\n\n"
                    constructed_str += f"## {descriptorsCombination} ({nb_directions} directions / {classifier.value.title()})"
                    
                    resultsDescriptorsCombination = results.scores[descriptorsCombination][classifier]
                    constructed_str += "\n\n| Final result | {:.2f} ± {:.2f} | ".format(np.mean(resultsDescriptorsCombination), np.std(resultsDescriptorsCombination))
                    constructed_str += "\n|---|---|"
                    constructed_str += "\n| Cross-validation results | {} |".format(" / ".join([str(round(crossResult, 4)) for crossResult in resultsDescriptorsCombination]))

                    matrix = results.matrices[descriptorsCombination][classifier]
                    matrix_md = self.__get_markdown_one_confusion_matrix(matrix, nb_directions)
                    constructed_str += "\n\n" + matrix_md

        return constructed_str


