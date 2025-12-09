
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

"""========================================================================
         Enhanced Genetic Grey lag goose optimization   
   ========================================================================"""  


def objective_function(selected_features, X_train, y_train, X_test, y_test):
    # Filter the features based on the selection
    X_train_selected = X_train[:, selected_features == 1]
    X_test_selected = X_test[:, selected_features == 1]
    # Train a model (RandomForest in this example)
    model = RandomForestClassifier()
    model.fit(X_train_selected, y_train)
    # Predict and calculate accuracy
    y_pred = model.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    # We minimize the negative accuracy to maximize accuracy
    return -accuracy


def initialize_population(pop_size, dimensions):
    return np.random.randint(0, 2, (pop_size, dimensions))


def update_parameters(a, t, t_max):
    r1, r2, r3, r4, r5 = np.random.random(5)
    a = 2 * (1 - t / t_max)
    A = 2 * a * r1 - a
    C = 2 * r2
    return A, C, r1, r2, r3, r4, r5


def exploration_update(X, pop_size, Indbest, srate, lrate, neighborhood, X_train, y_train, X_test, y_test):
    for i in range(pop_size):
        if np.random.rand() >= neighborhood:
            # Long distance movement
            M_I = lrate * np.random.rand() + objective_function(Indbest, X_train, y_train, X_test, y_test)
        else:
            # Short distance movement
            M_I = srate * np.random.rand() + objective_function(Indbest, X_train, y_train, X_test, y_test)
        rho = np.random.uniform(0.1, 1.0)  # Scale factor for displacement
        # Apply changes based on the calculated displacement
        for j in range(X.shape[1]):
            if np.random.rand() < rho * M_I:
                X[i, j] = 1 - X[i, j]  # Flip the bit (0 becomes 1, and 1 becomes 0)
    return X



def GGO(dimensions,features, encoded_labels,bounds=[0, 1], pop_size=30,t_max =10,srate =0.1,lrate=0.5):
    X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)
    # Initialize population and parameters
    X = initialize_population(pop_size, dimensions)
    P = X[np.argmin([objective_function(x, X_train, y_train, X_test, y_test) for x in X])]
    n1 = pop_size // 2
    n2 = pop_size - n1
    b, l, c = 1, 1, 1
    w, w1, w2, w3, w4 = np.random.random(5)
    A1, A2, A3 = np.random.random(3)
    C1, C2, C3 = np.random.random(3)
    D = np.random.uniform(0.1, 1.0)  # Displacement factor
    prev_f_values = np.array([objective_function(x, X_train, y_train, X_test, y_test) for x in X])
    for t in range(1, t_max + 1):
        A, C, r1, r2, r3, r4, r5 = update_parameters(2, t, t_max)
        z = 1 - (t / t_max) ** 2
        # Exploration group update using exploration_update function
        neighborhood = np.random.rand()  # Random neighborhood value for each iteration
        X[:n1] = exploration_update(X[:n1], n1, P, srate, lrate, neighborhood, X_train, y_train, X_test, y_test)
        # Exploitation group update
        for i in range(n1, pop_size):
            if t % 2 == 0:
                # Select three sentry positions
                indices = np.random.choice(np.arange(n1, pop_size), 3, replace=False)
                X_sentry1, X_sentry2, X_sentry3 = X[indices]
                X1 = X_sentry1 - A1 * np.abs(C1 * X_sentry1 - X[i])
                X2 = X_sentry2 - A2 * np.abs(C2 * X_sentry2 - X[i])
                X3 = X_sentry3 - A3 * np.abs(C3 * X_sentry3 - X[i])
                X[i] = (X1 + X2 + X3) / 3
            else:
                # Standard displacement update
                X_flock1 = X[np.random.choice(np.arange(n1, pop_size))]
                X[i] = X[i] + D * (1 + z) * w * (X[i] - X_flock1)
        # Calculate objective function values and update the best solution P
        f_values = np.array([objective_function(x, X_train, y_train, X_test, y_test) for x in X])
        best_index = np.argmin(f_values)
        if objective_function(X[best_index], X_train, y_train, X_test, y_test) < objective_function(P, X_train, y_train, X_test, y_test):
            P = X[best_index]
        # Adjust solutions if the best has not changed over two iterations
        if t > 1 and np.min(f_values) == np.min(prev_f_values):
            n1 = int(n1 * 1.1)
            n2 = pop_size - n1
            if n1 > pop_size:  # Ensure n1 doesn't exceed pop_size
                n1 = pop_size
                n2 = 0
        prev_f_values = f_values.copy()
    return P

