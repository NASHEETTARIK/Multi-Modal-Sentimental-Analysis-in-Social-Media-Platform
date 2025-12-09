import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

"""========================================================================
              Adapted Firefly Optimization Algorithm  
   ========================================================================"""   

# Define the objective function (accuracy of the RandomForestClassifier)
def objective_function(features, X, y):
    # Ensure that at least one feature is selected
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    # Select the features where the mask is 1
    selected_features = X[:, features == 1]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(selected_features, y, test_size=0.3, random_state=42)

    # Train a classifier and calculate fitness (accuracy)
    if X_train.shape[1] > 0:  # Ensure there's at least one feature to train on
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        return accuracy_score(y_test, predictions)
    else:
        return 0  # If no features are available after selection, return the lowest fitness score

# Distance between two fireflies
def euclidean_distance(x1, x2):
    return np.linalg.norm(x1 - x2)

# Move firefly i towards firefly j
def move_firefly(firefly_i, firefly_j, alpha, beta, gamma):
    r = euclidean_distance(firefly_i, firefly_j)
    beta = beta * np.exp(-gamma * r ** 2)
    
    # Update position (perform operations as float)
    firefly_i = firefly_i.astype(float)  # Ensure the array is float for calculations
    firefly_i += beta * (firefly_j - firefly_i) + alpha * (np.random.rand(len(firefly_i)) - 0.5)
    
    # Ensure binary solution (round to 0 or 1)
    firefly_i = np.clip(firefly_i, 0, 1)
    firefly_i = np.round(firefly_i).astype(int)  # Convert back to integers (0 or 1)
    
    return firefly_i

# Firefly Algorithm
def firefly_algorithm(X, y, n_fireflies, n_iter, feature_size, alpha, beta0, gamma):
    # Initialize fireflies randomly
    fireflies = np.random.randint(2, size=(n_fireflies, feature_size))
    fitness = np.zeros(n_fireflies)

    # Evaluate initial fitness
    for i in range(n_fireflies):
        fitness[i] = objective_function(fireflies[i], X, y)

    # Main loop of the Firefly Algorithm
    for t in range(n_iter):
        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if fitness[i] < fitness[j]:  # Firefly i is less bright than firefly j
                    fireflies[i] = move_firefly(fireflies[i], fireflies[j], alpha, beta0, gamma)
                    # Recalculate fitness after the move
                    fitness[i] = objective_function(fireflies[i], X, y)

        # Optional: Reduce randomness alpha over time
        alpha = alpha * 0.2

    # Find the best solution
    best_idx = np.argmax(fitness)
    return fireflies[best_idx], fitness[best_idx]

def firefly_algorithms(X,y):
    # Firefly Algorithm Hyperparameters
    n_fireflies = 20  # Number of fireflies
    n_iter = 10
    alpha = 0.5  # Randomness parameter
    beta0 = 1.0  # Attraction coefficient base value
    gamma = 1.0  # Light absorption coefficient

    feature_size = X.shape[1]

    # Run Firefly Algorithm
    best_solution, best_fitness = firefly_algorithm(X, y, n_fireflies, n_iter, feature_size, alpha, beta0, gamma)

    return best_solution