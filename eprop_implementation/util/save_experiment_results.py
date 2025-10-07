import json

def convert_dict_values_to_python_float(result):
    return {k: float(v) for k, v in result.items()}

def save_experiment_results(FLAGS, complete_time, overall_results, filename):

    overall_results = [convert_dict_values_to_python_float(result) for result in overall_results]

    result_dict = {
        **FLAGS,
        'complete_time': complete_time,
        'results': overall_results
    }

    with open(f"./results/{filename}.json", "w") as f:
        json.dump(result_dict, f)