import argparse
import pandas as pd
from models import linear_regression, random_forest, svr, base_nn, improved_nn, complex_nn

# Load files
log_file_path = './data/log.csv'
query_file_path = './data/query.csv'

log = pd.read_csv(log_file_path)
query = pd.read_csv(query_file_path)

Best_score = {}
Best_query = {}
Model_score = {}

# Data Description
log_A = log['A']
log_B = log['B']
log_C = log['C']
log_D = log['D']
log_E = log['E']
log_score = log['score']

n_1 = int(len(log['score']) * 0.1)
score_top_10 = log.nlargest(n_1, 'score')

model_functions = {
    'Linear Regression': linear_regression,
    'Random Forest': random_forest,
    'SVR': svr,
    'BaseNN': base_nn,
    'ImprovedNN': improved_nn,
    'ComplexNN': complex_nn
}

def main():
    parser = argparse.ArgumentParser(description='Testing models on dataset')
    parser.add_argument('-m', '--models', type=str, nargs='+', choices=['all'] + list(model_functions.keys()), help='List of models to evaluate or "all"')
    parser.add_argument('-o', '--option', type=str, nargs='+', default=['option1'], choices=['all', 'option1', 'option2'], help='Evaluation option(s) to use or "all"')
    
    args = parser.parse_args()

    # Determine models to evaluate
    if 'all' in args.models:
        models_to_evaluate = list(model_functions.keys())
    else:
        models_to_evaluate = args.models

    # Determine options to use
    if 'all' in args.option:
        options_to_use = ['option1', 'option2']
    else:
        options_to_use = args.option

    # Evaluate each selected model with each selected option
    global Best_score, Best_query, Model_score

    for option in options_to_use:
        for model_name in models_to_evaluate:
            if model_name in model_functions:
                print(f"Evaluating {model_name} with {option}")
                if option == 'option1':
                    input_data = log
                elif option == 'option2':
                    input_data = score_top_10
                try : 
                    Best_score, Best_query, Model_score = model_functions[model_name](input_data, Best_score, Best_query, Model_score, option)
                except : 
                    print(f'Testing model {model_name} stopped for some reason.')                    
                    pass

    # Print best model results
    if Best_score:
        max_value_key = max(Best_score, key=lambda k: Best_score[k])
        print(f'The Model {max_value_key} was the best model\n')
        df = pd.DataFrame(Best_query[max_value_key], columns=["A", "B", "C", "D", "E"])
        print(f'The Query of model {max_value_key} is')
        print(df)
        print(f'The Model Score : {Model_score}')
    else:
        print("No models were evaluated. Please check the input parameters.")

if __name__ == "__main__":
    main()
