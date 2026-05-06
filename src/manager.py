import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from .dataComputing.ComputeDescriptorsFromDatabase import ComputeDescriptorsFromDatabase
from .dataComputing.ComputeResults import ComputeResults
from .dataComputing.SaveAndDisplayResults import SaveAndDisplayResults
from .descriptors.utils.DescriptorsParameters import DescriptorsParameters
from .utils.Parameters import Parameters
from .utils.enumerations import ReferencePointMode


def _compute_metrics(y_true: list, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _majority_vote(run_predictions: list[np.ndarray]) -> np.ndarray:
    """Majority vote per sample over a list of prediction arrays (one array per run)."""
    if len(run_predictions) == 0:
        return np.array([])

    stacked = np.stack(run_predictions, axis=0)
    vote = []

    for col in stacked.T:
        labels, counts = np.unique(col, return_counts=True)
        max_count = np.max(counts)
        winners = set(labels[counts == max_count])

        # Tie-break: first run's prediction among the tied winners.
        chosen = None
        for candidate in col:
            if candidate in winners:
                chosen = candidate
                break
        vote.append(chosen)

    return np.array(vote)


def _run_legacy_single_experiment(parameters: Parameters):
    """Original single-run behavior (kept for backward compatibility)."""
    global_results = SaveAndDisplayResults()
    descriptors_parameters = DescriptorsParameters(parameters)

    for nb_directions in parameters.nb_directions:
        descriptors_computing = ComputeDescriptorsFromDatabase(parameters.database)
        descriptors_computing.compute_descriptors(nb_directions, descriptors_parameters)

        results_computing = ComputeResults()
        results_computing.compute_results_from_descriptors_combinations_and_classifiers(
            nb_directions=nb_directions,
            computedDescriptors=descriptors_computing,
            parameters=parameters,
            random_seed=parameters.random_seed,
        )

        global_results.insert_results(nb_directions, results_computing)

    global_results.display_results()
    global_results.write_results_in_file(parameters.database)


def _run_multi_iterations_for_mode(parameters: Parameters, rp_mode: ReferencePointMode) -> dict:
    descriptors_parameters = DescriptorsParameters(parameters)
    descriptors_parameters.rp_mode = rp_mode

    aggregated = {nb_dir: dict() for nb_dir in parameters.nb_directions}
    run_rp_points = []

    for run_idx in range(parameters.nb_iterations):
        iteration_seed = parameters.random_seed + run_idx
        descriptors_parameters.reset_rp_for_iteration(iteration_seed)
        selected_point_this_run = None

        print(f"\n=== Run {run_idx + 1}/{parameters.nb_iterations} ({rp_mode.value}, seed={iteration_seed}) ===")

        for nb_directions in parameters.nb_directions:
            descriptors_computing = ComputeDescriptorsFromDatabase(parameters.database)
            descriptors_computing.compute_descriptors(nb_directions, descriptors_parameters)
            if selected_point_this_run is None:
                selected_point_this_run = descriptors_parameters.rp_selected_point

            results_computing = ComputeResults()
            results_computing.compute_results_from_descriptors_combinations_and_classifiers(
                nb_directions=nb_directions,
                computedDescriptors=descriptors_computing,
                parameters=parameters,
                random_seed=iteration_seed,
            )

            labels = (
                list(descriptors_computing.labels4directions)
                if nb_directions == 4
                else list(descriptors_computing.labels8directions)
            )

            for key_descriptor_combination in results_computing.scores.keys():
                aggregated[nb_directions].setdefault(key_descriptor_combination, dict())

                for classifier in parameters.classifiers:
                    aggregated[nb_directions][key_descriptor_combination].setdefault(
                        classifier,
                        {
                            "run_scores": [],
                            "run_metrics": [],
                            "run_predictions": [],
                            "y_true": np.array(descriptors_computing.Y_data),
                            "labels": labels,
                        },
                    )

                    current_data = aggregated[nb_directions][key_descriptor_combination][classifier]
                    current_data["run_scores"].append(results_computing.scores[key_descriptor_combination][classifier])
                    current_data["run_metrics"].append(results_computing.metrics[key_descriptor_combination][classifier])
                    current_data["run_predictions"].append(results_computing.predictions[key_descriptor_combination][classifier])

        run_rp_points.append(selected_point_this_run)

    for nb_directions in aggregated.keys():
        for key_descriptor_combination in aggregated[nb_directions].keys():
            for classifier in aggregated[nb_directions][key_descriptor_combination].keys():
                current_data = aggregated[nb_directions][key_descriptor_combination][classifier]

                vote_predictions = _majority_vote(current_data["run_predictions"])
                vote_metrics = _compute_metrics(current_data["y_true"], vote_predictions)
                vote_matrix = confusion_matrix(
                    current_data["y_true"], vote_predictions, labels=current_data["labels"]
                )

                current_data["vote_predictions"] = vote_predictions
                current_data["vote_metrics"] = vote_metrics
                current_data["vote_matrix"] = vote_matrix

                metric_names = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
                current_data["mean_metrics"] = {
                    metric: float(np.mean([run_metric[metric] for run_metric in current_data["run_metrics"]]))
                    for metric in metric_names
                }
                current_data["std_metrics"] = {
                    metric: float(np.std([run_metric[metric] for run_metric in current_data["run_metrics"]]))
                    for metric in metric_names
                }

    return {
        "rp_mode": rp_mode,
        "run_rp_points": run_rp_points,
        "aggregated": aggregated,
    }


def _display_multi_iteration_results(summary: dict, parameters: Parameters):
    rp_mode = summary["rp_mode"]
    aggregated = summary["aggregated"]

    print(f"\n\n========== Results for Rp mode: {rp_mode.value} ==========")

    if rp_mode != ReferencePointMode.DETERMINISTIC:
        print("Rp sampled once per run (then reused for all images):")
        for i, point in enumerate(summary["run_rp_points"]):
            print(f"\t- Run {i + 1}: Rp={point}")

    for nb_directions in parameters.nb_directions:
        print(f"\n• Results for {nb_directions} directions:")
        for key_descriptor_combination in aggregated[nb_directions].keys():
            print(f"\n\t• Descriptors: {key_descriptor_combination}")

            for classifier in parameters.classifiers:
                current_data = aggregated[nb_directions][key_descriptor_combination][classifier]
                print(f"\t\t- Classifier: {classifier.value}")

                for run_idx, (run_scores, run_metrics) in enumerate(
                    zip(current_data["run_scores"], current_data["run_metrics"])
                ):
                    print(
                        "\t\t  Run {:>2}: cv={:.3f} ± {:.3f} | acc={:.3f} | p={:.3f} | r={:.3f} | f1={:.3f}".format(
                            run_idx + 1,
                            float(np.mean(run_scores)),
                            float(np.std(run_scores)),
                            run_metrics["accuracy"],
                            run_metrics["precision_macro"],
                            run_metrics["recall_macro"],
                            run_metrics["f1_macro"],
                        )
                    )

                mean_metrics = current_data["mean_metrics"]
                std_metrics = current_data["std_metrics"]
                vote_metrics = current_data["vote_metrics"]
                print(
                    "\t\t  Mean over runs: acc={:.3f}±{:.3f} | p={:.3f}±{:.3f} | r={:.3f}±{:.3f} | f1={:.3f}±{:.3f}".format(
                        mean_metrics["accuracy"],
                        std_metrics["accuracy"],
                        mean_metrics["precision_macro"],
                        std_metrics["precision_macro"],
                        mean_metrics["recall_macro"],
                        std_metrics["recall_macro"],
                        mean_metrics["f1_macro"],
                        std_metrics["f1_macro"],
                    )
                )
                print(
                    "\t\t  Majority vote: acc={:.3f} | p={:.3f} | r={:.3f} | f1={:.3f}".format(
                        vote_metrics["accuracy"],
                        vote_metrics["precision_macro"],
                        vote_metrics["recall_macro"],
                        vote_metrics["f1_macro"],
                    )
                )


def _display_comparison_against_deterministic(
    summary_random: dict, summary_deterministic: dict, parameters: Parameters
):
    print("\n\n========== Comparison vs deterministic Rp ==========")

    aggregated_random = summary_random["aggregated"]
    aggregated_deterministic = summary_deterministic["aggregated"]

    for nb_directions in parameters.nb_directions:
        print(f"\n• {nb_directions} directions:")
        for key_descriptor_combination in aggregated_random[nb_directions].keys():
            print(f"\n\t• Descriptors: {key_descriptor_combination}")
            for classifier in parameters.classifiers:
                random_data = aggregated_random[nb_directions][key_descriptor_combination][classifier]
                deterministic_data = aggregated_deterministic[nb_directions][key_descriptor_combination][classifier]

                random_vote_acc = random_data["vote_metrics"]["accuracy"]
                deterministic_vote_acc = deterministic_data["vote_metrics"]["accuracy"]
                delta_vote_acc = random_vote_acc - deterministic_vote_acc

                random_mean_acc = random_data["mean_metrics"]["accuracy"]
                deterministic_mean_acc = deterministic_data["mean_metrics"]["accuracy"]
                delta_mean_acc = random_mean_acc - deterministic_mean_acc

                print(
                    "\t\t- {}: Δ(mean acc)={:+.3f}, Δ(vote acc)={:+.3f} ({} vs {})".format(
                        classifier.value,
                        delta_mean_acc,
                        delta_vote_acc,
                        summary_random["rp_mode"].value,
                        ReferencePointMode.DETERMINISTIC.value,
                    )
                )


def manage(parameters: Parameters):
    show_parameters(parameters)

    # Keep the previous behavior unchanged for the default historical setting.
    if (
        parameters.nb_iterations == 1
        and parameters.rp_mode == ReferencePointMode.DETERMINISTIC
    ):
        _run_legacy_single_experiment(parameters)
        return

    rp_modes_to_run = [parameters.rp_mode]
    if parameters.rp_mode != ReferencePointMode.DETERMINISTIC:
        rp_modes_to_run.append(ReferencePointMode.DETERMINISTIC)

    summaries = dict()
    for rp_mode in rp_modes_to_run:
        summary = _run_multi_iterations_for_mode(parameters, rp_mode)
        summaries[rp_mode] = summary
        _display_multi_iteration_results(summary, parameters)

    if (
        parameters.rp_mode != ReferencePointMode.DETERMINISTIC
        and ReferencePointMode.DETERMINISTIC in summaries
    ):
        _display_comparison_against_deterministic(
            summaries[parameters.rp_mode], summaries[ReferencePointMode.DETERMINISTIC], parameters
        )


def show_parameters(parameters: Parameters):
    """Show the parameters, so that we know which ones will be computed."""
    print("Parameters :")
    print(f"\t• Database : {parameters.database.value}")
    print("\t• Classifier(s) : {}".format(", ".join([c.value for c in parameters.classifiers])))
    print("\t• Nb of directions : {}".format(", ".join([str(nb_dir) for nb_dir in parameters.nb_directions])))

    get_str_descriptor = lambda descriptor: descriptor if type(descriptor) is str else descriptor.value
    get_str_combination = lambda list_descriptors: [get_str_descriptor(descriptor) for descriptor in list_descriptors]
    sub_str_list = ", ".join([str(get_str_combination(sub_list)) for sub_list in parameters.descriptors_layout])
    print("\t• Descriptors : {}".format(str(sub_str_list)))

    print(f"\t• Rp mode : {parameters.rp_mode.value}")
    if parameters.rp_mode != ReferencePointMode.DETERMINISTIC:
        print(f"\t• Rp border size (px) : {parameters.rp_border_size}")
    print(f"\t• Iterations : {parameters.nb_iterations}")
    print(f"\t• Base random seed : {parameters.random_seed}")
    if parameters.rp_mode != ReferencePointMode.DETERMINISTIC:
        print("\t• Deterministic baseline : enabled (computed automatically for comparison)")

    print()
