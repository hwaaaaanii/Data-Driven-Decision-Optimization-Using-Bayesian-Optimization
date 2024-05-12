# Libraries and file path
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

from bayes_opt import BayesianOptimization

from DLmodels import BaseMLP, ImprovedMLP, ComplexMLP

import warnings
warnings.filterwarnings("ignore", category=UserWarning)



def linear_regression(log, Best_score, Best_query, Model_score, option = 'option1'):

    X_regression = log[['A', 'B', 'C', 'D', 'E']]
    y_regression = log['score']

    if option == 'option1':
        label = 'Raw Data'
    else: 
        label = 'Top 10 Percent Data'

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_regression, y_regression, test_size=0.2, random_state=42)

    # Training the linear regression model
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # Predicting 'score' for the test set
    y_pred_test = linear_model.predict(X_test)

    # Predicting 'score' for the entire dataset
    predicted_scores = linear_model.predict(X_regression)
    X_regression['predicted_score'] = predicted_scores

    # Extracting the top 10 rows with the highest predicted 'score'
    top_predictions_regression = X_regression.nlargest(10, 'predicted_score')

    # Displaying the top 10 predictions
    print(top_predictions_regression[['A', 'B', 'C', 'D', 'E', 'predicted_score']])

    # Calculating the average of the top 10 predicted 'scores'
    average_score = top_predictions_regression['predicted_score'].mean()
    Best_score[f'Linear Regression for {label}'] = average_score
    Best_query[f'Linear Regression for {label}'] = top_predictions_regression[['A', 'B', 'C', 'D', 'E', 'predicted_score']]

    print(f"\nAverage score of top 10 predictions using Linear Regression for {label} :", average_score)

    # Calculating and displaying the MSE and R^2 for the test set
    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    print("Mean Squared Error (MSE) on Test Set :", mse)
    print("R^2 Score on Test Set :", r2)
    model_score = r2 * (2 * average_score * (1 / np.sqrt(mse))) / (average_score + (1 / np.sqrt(mse)))
    print(f'Model Score : {model_score}')
    Model_score[f'Linear Regression for {label}'] = model_score

    print('-----------------------Linear Regression Process Done-----------------------')
    print('----------------------------------------------------------------------------\n\n')

    return Best_score, Best_query, Model_score




def random_forest(log, Best_score, Best_query, Model_score, option = 'option1'):

    X = log[['A', 'B', 'C', 'D', 'E']]
    y = log['score']

    if option == 'option1':
        label = 'Raw Data'
    else: 
        label = 'Top 10 Percent Data'

    print('----------------------------------------------------------------------------')
    print('---------------------------Processing Random Forest-------------------------')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Predicting 'score' for the test set using the trained Random Forest model
    y_pred_test_rf = rf_model.predict(X_test)

    # Calculating MSE and R^2 for the test set predictions
    mse_rf = mean_squared_error(y_test, y_pred_test_rf)
    r2_rf = r2_score(y_test, y_pred_test_rf)

    # Objective function to maximize using Bayesian Optimization
    def rf_score(A, B, C, D, E):
        X_new = [[A, B, C, D, E]]
        return rf_model.predict(X_new)[0]

    # Parameter bounds for Bayesian Optimization
    pbounds = {'A': (0, 1), 'B': (0, 1), 'C': (0, 1), 'D': (0, 1), 'E': (0, 1)}

    print('------------Finding Optimal Points using Bayesian Optimization------------')

    # Bayesian Optimization object creation
    optimizer = BayesianOptimization(
        f=rf_score,
        pbounds=pbounds,
        verbose=1,
        random_state=42
    )

    # Optimization execution
    optimizer.maximize(
        init_points=5,
        n_iter=200
    )

    # Extracting the top 10 results based on the target (predicted score)
    top_results = sorted(optimizer.res, key=lambda x: x['target'], reverse=True)[:10]

    # Preparing the DataFrame to display the top results
    df = {
        "A": [],
        "B": [],
        "C": [],
        "D": [],
        "E": [],
        "score": []
    }

    for result in top_results:
        df['A'].append(result['params']['A'])
        df['B'].append(result['params']['B'])
        df['C'].append(result['params']['C'])
        df['D'].append(result['params']['D'])
        df['E'].append(result['params']['E'])
        df['score'].append(result['target'])

    top_predictions_rf = pd.DataFrame(df)
    print(top_predictions_rf)

    # Calculating the average score of the top 10 predictions
    average_score = np.mean(top_predictions_rf['score'])
    Best_score[f'Random Forest for {label}'] = average_score
    Best_query[f'Random Forest for {label}'] = top_predictions_rf

    print("\nAverage score of top 10 predictions using Random Forest: ", average_score)
    print(f"Mean Squared Error (MSE) on Test Set: {mse_rf}")
    print(f"R^2 Score on Test Set: {r2_rf}")
    model_score = r2_rf * (2 * average_score * (1 / np.sqrt(mse_rf))) / (average_score + (1 / np.sqrt(mse_rf)))
    print(f'Model Score : {model_score}')
    Model_score[f'Random Forest for {label}'] = model_score

    print('-------------------------Random Forest Process Done-------------------------')
    print('----------------------------------------------------------------------------\n\n')

    return Best_score, Best_query, Model_score




def svr(log, Best_score, Best_query, Model_score, option = 'option1'):

    X = log[['A', 'B', 'C', 'D', 'E']]
    y = log['score']

    if option == 'option1':
        label = 'Raw Data'
    else: 
        label = 'Top 10 Percent Data'

    print('----------------------------------------------------------------------------')
    print('---------------------Processing Support Vector Regression-------------------')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training the Support Vector Regression (SVR) model with chosen parameters
    model = SVR(kernel='rbf', C=195, epsilon=0.1)
    model.fit(X_train, y_train)

    # Predicting 'score' for the test set using the trained SVR model
    y_pred_test_svr = model.predict(X_test)

    # Objective function to maximize using Bayesian Optimization
    def svr_score(A, B, C, D, E):
        X_new = [[A, B, C, D, E]]
        return model.predict(X_new)[0]

    # Parameter bounds for Bayesian Optimization
    pbounds = {'A': (0, 1), 'B': (0, 1), 'C': (0, 1), 'D': (0, 1), 'E': (0, 1)}

    print('------------Finding Optimal Points using Bayesian Optimization------------')

    # Bayesian Optimization object creation
    optimizer = BayesianOptimization(
        f=svr_score,
        pbounds=pbounds,
        verbose=1,
        random_state=42
    )

    # Optimization execution
    optimizer.maximize(
        init_points=5,
        n_iter=200
    )

    # Extracting the top 10 results based on the target (predicted score)
    top_results = sorted(optimizer.res, key=lambda x: x['target'], reverse=True)[:10]

    # Preparing the DataFrame to display the top results
    df = {
        "A": [],
        "B": [],
        "C": [],
        "D": [],
        "E": [],
        "score": []
    }

    for result in top_results:
        df['A'].append(result['params']['A'])
        df['B'].append(result['params']['B'])
        df['C'].append(result['params']['C'])
        df['D'].append(result['params']['D'])
        df['E'].append(result['params']['E'])
        df['score'].append(result['target'])

    top_predictions_svr = pd.DataFrame(df)
    print(top_predictions_svr)

    # Calculating the average score of the top 10 predictions
    average_score = np.mean(top_predictions_svr['score'])
    Best_score[f'SVR for {label}'] = average_score
    Best_query[f'SVR for {label}'] = top_predictions_svr

    print(f"\nAverage score of top 10 predictions using SVR: {average_score}")

    # Calculating MSE and R^2 for the test set predictions
    mse_svr = mean_squared_error(y_test, y_pred_test_svr)
    r2_svr = r2_score(y_test, y_pred_test_svr)

    print(f"Mean Squared Error (MSE) on Test Set: {mse_svr}")
    print(f"R^2 Score on Test Set: {r2_svr}")
    model_score = r2_svr * (2 * average_score * (1 / np.sqrt(mse_svr))) / (average_score + (1 / np.sqrt(mse_svr)))
    print(f'Model Score : {model_score}')
    Model_score[f'SVR for {label}'] = model_score

    print('-------------------Support Vector Regression Process Done-------------------')
    print('----------------------------------------------------------------------------\n\n')

    return Best_score, Best_query, Model_score





def base_nn(log, Best_score, Best_query, Model_score, option = 'option1'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_selection = 'Base'
    n_1, p = int(len(log['score']) * 0.1), 5

    data = torch.from_numpy(log.values).float().to(device)
    X, y = data[:, :5], data[:, -1:]

    if option == 'option1':
        label = 'Raw Data'

    else: 
        label = 'Top 10 Percent Data'

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

    optimizer_selection = 'Adam'

    batch_size = 256

    if model_selection == 'Base':
        training_step = 50000
    else:
        training_step = 500000

    validation_interval = 2000
    lr = 1e-4

    criterion = nn.MSELoss()

    if model_selection == 'Base':
        model = BaseMLP(input_dim=5, output_dim=1, hidden_dim=64)
    elif model_selection == 'Improved':
        model = ImprovedMLP(input_dim=5, output_dim=1, hidden_dim=64)
    elif model_selection == 'Complex':
        model = ComplexMLP(input_dim=5, output_dim=1, hidden_dims=[128, 256, 64])

    if optimizer_selection == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_selection == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr)

    print(f'---------------------Training {model_selection} Neural Network-------------------') 
    print(f'Device : {device}')

    model = model.to(device)
    train_losses = []
    valid_losses = []

    for step in range(training_step):
        idx = torch.randint(0, len(X_train_tensor), size=(batch_size,))
        batch_train_x, batch_train_y = X_train_tensor[idx], y_train_tensor[idx]
        batch_pred_y = model(batch_train_x)

        optimizer.zero_grad()
        train_loss = criterion(batch_train_y, batch_pred_y)
        train_loss.backward()
        optimizer.step()

        train_losses.append(train_loss.item())

        if (step + 1) % validation_interval == 0:
            valid_loss = criterion(model(X_val_tensor), y_val_tensor)
            valid_losses.append(valid_loss.item())
            print(f"Step: {step + 1}/{training_step}\tTrain Loss: {train_losses[-1]:.2f}\tValid Loss: {valid_losses[-1]:.2f}")

    print('-----------------------------Experiment setting------------------------------')
    print(f'Model : {model}')
    print(f'Optimizer : {optimizer}')
    print(f'Batch size : {batch_size}')
    print(f'Training Step : {training_step}')
    print(f'Learning rate : {lr}')
    print(' ')

    model.eval()
    with torch.no_grad():
        test_pred_y = model(X_test_tensor)
        test_loss = criterion(test_pred_y, y_test_tensor.view(-1, 1))

    test_mse = test_loss.item()
    test_mae = torch.mean(torch.abs(test_pred_y - y_test_tensor)).item()
    ss_tot = ((y_test_tensor - torch.mean(y_test_tensor)) ** 2).sum()
    ss_res = ((y_test_tensor - test_pred_y) **2).sum()
    test_r2 = 1 - ss_res / ss_tot
    test_r2 = 1 - (1 - test_r2) * (n_1 - 1) / (n_1 - 5 - 1)

    def rf_score(A, B, C, D, E):
        X_new = [[A, B, C, D, E]]
        X_new = torch.tensor(X_new, dtype=torch.float32)
                
        if X_new.dim() == 1:
            X_new = X_new.unsqueeze(0)

        model.eval()
        return model(X_new).item()

    pbounds = {'A': (0, 1), 'B': (0, 1), 'C': (0, 1), 'D': (0, 1), 'E': (0, 1)}

    print('------------Finding Optimal Points using Bayesian Optimization------------')

    optimizer = BayesianOptimization(
        f=rf_score,
        pbounds=pbounds,
        verbose=1,
        random_state=1,
    )

    num_iter = 200

    optimizer.maximize(
        init_points=5,
        n_iter=num_iter
    )

    top_results = sorted(optimizer.res, key=lambda x: x['target'], reverse=True)[:10]

    df = {
        "A": [],
        "B": [],
        "C": [],
        "D": [],
        "E": [],
        "score": []
    }

    for result in top_results:
        df['A'].append(result['params']['A'])
        df['B'].append(result['params']['B'])
        df['C'].append(result['params']['C'])
        df['D'].append(result['params']['D'])
        df['E'].append(result['params']['E'])
        df['score'].append(result['target'])

    top_predictions_rf = pd.DataFrame(df)
    print(top_predictions_rf)

    average_score = np.mean(top_predictions_rf['score'])
    Best_score[f'{model_selection} Neural Network for {label}'] = average_score
    Best_query[f'{model_selection} Neural Network for {label}'] = top_predictions_rf

    print(f"\nAverage score of top 10 predictions using {model_selection} Neural Network : ", average_score)
    print(f"Mean Squared Error (MSE) on Test Set : {test_mse}")
    print(f"R^2 Score on Test Set : {test_r2}")
    model_score = test_r2 * (2 * average_score * (1 / np.sqrt(test_mse))) / (average_score + (1 / np.sqrt(test_mse)))
    print(f'Model Score : {model_score}')
    Model_score[f'{model_selection} Neural Network for {label}'] = model_score

    print(f'---------------------{model_selection} Neural Network Process Done---------------------')
    print('----------------------------------------------------------------------------\n\n\n\n')    

    return Best_score, Best_query, Model_score






def improved_nn(log, Best_score, Best_query, Model_score, option = 'option1'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_selection = 'Improved'
    n_1, p = int(len(log['score']) * 0.1), 5

    data = torch.from_numpy(log.values).float().to(device)
    X, y = data[:, :5], data[:, -1:]

    if option == 'option1':
        label = 'Raw Data'

    else: 
        label = 'Top 10 Percent Data'

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

    optimizer_selection = 'Adam'

    batch_size = 256

    if model_selection == 'Base':
        training_step = 50000
    else:
        training_step = 500000

    validation_interval = 2000
    lr = 1e-4

    criterion = nn.MSELoss()

    if model_selection == 'Base':
        model = BaseMLP(input_dim=5, output_dim=1, hidden_dim=64)
    elif model_selection == 'Improved':
        model = ImprovedMLP(input_dim=5, output_dim=1, hidden_dim=64)
    elif model_selection == 'Complex':
        model = ComplexMLP(input_dim=5, output_dim=1, hidden_dims=[128, 256, 64])

    if optimizer_selection == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_selection == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr)

    print(f'---------------------Training {model_selection} Neural Network-------------------') 
    print(f'Device : {device}')

    model = model.to(device)
    train_losses = []
    valid_losses = []

    for step in range(training_step):
        idx = torch.randint(0, len(X_train_tensor), size=(batch_size,))
        batch_train_x, batch_train_y = X_train_tensor[idx], y_train_tensor[idx]
        batch_pred_y = model(batch_train_x)

        optimizer.zero_grad()
        train_loss = criterion(batch_train_y, batch_pred_y)
        train_loss.backward()
        optimizer.step()

        train_losses.append(train_loss.item())

        if (step + 1) % validation_interval == 0:
            valid_loss = criterion(model(X_val_tensor), y_val_tensor)
            valid_losses.append(valid_loss.item())
            print(f"Step: {step + 1}/{training_step}\tTrain Loss: {train_losses[-1]:.2f}\tValid Loss: {valid_losses[-1]:.2f}")

    print('-----------------------------Experiment setting------------------------------')
    print(f'Model : {model}')
    print(f'Optimizer : {optimizer}')
    print(f'Batch size : {batch_size}')
    print(f'Training Step : {training_step}')
    print(f'Learning rate : {lr}')
    print(' ')

    model.eval()
    with torch.no_grad():
        test_pred_y = model(X_test_tensor)
        test_loss = criterion(test_pred_y, y_test_tensor.view(-1, 1))

    test_mse = test_loss.item()
    test_mae = torch.mean(torch.abs(test_pred_y - y_test_tensor)).item()
    ss_tot = ((y_test_tensor - torch.mean(y_test_tensor)) ** 2).sum()
    ss_res = ((y_test_tensor - test_pred_y) **2).sum()
    test_r2 = 1 - ss_res / ss_tot
    test_r2 = 1 - (1 - test_r2) * (n_1 - 1) / (n_1 - 5 - 1)

    def rf_score(A, B, C, D, E):
        X_new = [[A, B, C, D, E]]
        X_new = torch.tensor(X_new, dtype=torch.float32)
                
        if X_new.dim() == 1:
            X_new = X_new.unsqueeze(0)

        model.eval()
        return model(X_new).item()

    pbounds = {'A': (0, 1), 'B': (0, 1), 'C': (0, 1), 'D': (0, 1), 'E': (0, 1)}

    print('------------Finding Optimal Points using Bayesian Optimization------------')

    optimizer = BayesianOptimization(
        f=rf_score,
        pbounds=pbounds,
        verbose=1,
        random_state=1,
    )

    num_iter = 200

    optimizer.maximize(
        init_points=5,
        n_iter=num_iter
    )

    top_results = sorted(optimizer.res, key=lambda x: x['target'], reverse=True)[:10]

    df = {
        "A": [],
        "B": [],
        "C": [],
        "D": [],
        "E": [],
        "score": []
    }

    for result in top_results:
        df['A'].append(result['params']['A'])
        df['B'].append(result['params']['B'])
        df['C'].append(result['params']['C'])
        df['D'].append(result['params']['D'])
        df['E'].append(result['params']['E'])
        df['score'].append(result['target'])

    top_predictions_rf = pd.DataFrame(df)
    print(top_predictions_rf)

    average_score = np.mean(top_predictions_rf['score'])
    Best_score[f'{model_selection} Neural Network for {label}'] = average_score
    Best_query[f'{model_selection} Neural Network for {label}'] = top_predictions_rf

    print(f"\nAverage score of top 10 predictions using {model_selection} Neural Network : ", average_score)
    print(f"Mean Squared Error (MSE) on Test Set : {test_mse}")
    print(f"R^2 Score on Test Set : {test_r2}")
    model_score = test_r2 * (2 * average_score * (1 / np.sqrt(test_mse))) / (average_score + (1 / np.sqrt(test_mse)))
    print(f'Model Score : {model_score}')
    Model_score[f'{model_selection} Neural Network for {label}'] = model_score

    print(f'---------------------{model_selection} Neural Network Process Done---------------------')
    print('----------------------------------------------------------------------------\n\n\n\n')  

    return Best_score, Best_query, Model_score




def complex_nn(log, Best_score, Best_query, Model_score, option = 'option1'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_selection = 'Complex'
    n_1, p = int(len(log['score']) * 0.1), 5

    data = torch.from_numpy(log.values).float().to(device)
    X, y = data[:, :5], data[:, -1:]

    if option == 'option1':
        label = 'Raw Data'

    else: 
        label = 'Top 10 Percent Data'

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

    optimizer_selection = 'Adam'

    batch_size = 256

    if model_selection == 'Base':
        training_step = 50000
    else:
        training_step = 500000

    validation_interval = 2000
    lr = 1e-4

    criterion = nn.MSELoss()

    if model_selection == 'Base':
        model = BaseMLP(input_dim=5, output_dim=1, hidden_dim=64)
    elif model_selection == 'Improved':
        model = ImprovedMLP(input_dim=5, output_dim=1, hidden_dim=64)
    elif model_selection == 'Complex':
        model = ComplexMLP(input_dim=5, output_dim=1, hidden_dims=[128, 256, 64])

    if optimizer_selection == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_selection == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr)

    print(f'---------------------Training {model_selection} Neural Network-------------------') 
    print(f'Device : {device}')

    model = model.to(device)
    train_losses = []
    valid_losses = []

    for step in range(training_step):
        idx = torch.randint(0, len(X_train_tensor), size=(batch_size,))
        batch_train_x, batch_train_y = X_train_tensor[idx], y_train_tensor[idx]
        batch_pred_y = model(batch_train_x)

        optimizer.zero_grad()
        train_loss = criterion(batch_train_y, batch_pred_y)
        train_loss.backward()
        optimizer.step()

        train_losses.append(train_loss.item())

        if (step + 1) % validation_interval == 0:
            valid_loss = criterion(model(X_val_tensor), y_val_tensor)
            valid_losses.append(valid_loss.item())
            print(f"Step: {step + 1}/{training_step}\tTrain Loss: {train_losses[-1]:.2f}\tValid Loss: {valid_losses[-1]:.2f}")

    print('-----------------------------Experiment setting------------------------------')
    print(f'Model : {model}')
    print(f'Optimizer : {optimizer}')
    print(f'Batch size : {batch_size}')
    print(f'Training Step : {training_step}')
    print(f'Learning rate : {lr}')
    print(' ')

    model.eval()
    with torch.no_grad():
        test_pred_y = model(X_test_tensor)
        test_loss = criterion(test_pred_y, y_test_tensor.view(-1, 1))

    test_mse = test_loss.item()
    test_mae = torch.mean(torch.abs(test_pred_y - y_test_tensor)).item()
    ss_tot = ((y_test_tensor - torch.mean(y_test_tensor)) ** 2).sum()
    ss_res = ((y_test_tensor - test_pred_y) **2).sum()
    test_r2 = 1 - ss_res / ss_tot
    test_r2 = 1 - (1 - test_r2) * (n_1 - 1) / (n_1 - 5 - 1)

    def rf_score(A, B, C, D, E):
        X_new = [[A, B, C, D, E]]
        X_new = torch.tensor(X_new, dtype=torch.float32)
                
        if X_new.dim() == 1:
            X_new = X_new.unsqueeze(0)

        model.eval()
        return model(X_new).item()

    pbounds = {'A': (0, 1), 'B': (0, 1), 'C': (0, 1), 'D': (0, 1), 'E': (0, 1)}

    print('------------Finding Optimal Points using Bayesian Optimization------------')

    optimizer = BayesianOptimization(
        f=rf_score,
        pbounds=pbounds,
        verbose=1,
        random_state=1,
    )

    num_iter = 200

    optimizer.maximize(
        init_points=5,
        n_iter=num_iter
    )

    top_results = sorted(optimizer.res, key=lambda x: x['target'], reverse=True)[:10]

    df = {
        "A": [],
        "B": [],
        "C": [],
        "D": [],
        "E": [],
        "score": []
    }

    for result in top_results:
        df['A'].append(result['params']['A'])
        df['B'].append(result['params']['B'])
        df['C'].append(result['params']['C'])
        df['D'].append(result['params']['D'])
        df['E'].append(result['params']['E'])
        df['score'].append(result['target'])

    top_predictions_rf = pd.DataFrame(df)
    print(top_predictions_rf)

    average_score = np.mean(top_predictions_rf['score'])
    Best_score[f'{model_selection} Neural Network for {label}'] = average_score
    Best_query[f'{model_selection} Neural Network for {label}'] = top_predictions_rf

    print(f"\nAverage score of top 10 predictions using {model_selection} Neural Network : ", average_score)
    print(f"Mean Squared Error (MSE) on Test Set : {test_mse}")
    print(f"R^2 Score on Test Set : {test_r2}")
    model_score = test_r2 * (2 * average_score * (1 / np.sqrt(test_mse))) / (average_score + (1 / np.sqrt(test_mse)))
    print(f'Model Score : {model_score}')
    Model_score[f'{model_selection} Neural Network for {label}'] = model_score

    print(f'---------------------{model_selection} Neural Network Process Done---------------------')
    print('----------------------------------------------------------------------------\n\n\n\n')  

    return Best_score, Best_query, Model_score
