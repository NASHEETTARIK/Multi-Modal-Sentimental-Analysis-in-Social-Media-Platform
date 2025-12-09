import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

"""========================================================================
                 Parrot optimization algorithm   
   ========================================================================"""  



class ParrotOptimizer:
    def __init__(self, n_features, n_parrots, max_iter):
        self.n_features = n_features
        self.n_parrots = n_parrots
        self.max_iter = max_iter
        self.positions = np.random.rand(n_parrots, n_features)  # Initialize positions randomly
        self.best_positions = np.copy(self.positions)  # Store best positions
        self.best_fitness = np.inf  # Initialize best fitness

    def fitness(self, position, X_train, y_train):
        # Convert the position to a feature subset
        features = [i for i, val in enumerate(position) if val > 0.5]
        if not features:
            return 0  # Return zero fitness if no features are selected
        X_subset = X_train[:, features]
        # Train a classifier and calculate the accuracy
        clf = RandomForestClassifier()
        clf.fit(X_subset, y_train)
        y_pred = clf.predict(X_subset)
        return accuracy_score(y_train, y_pred)

    def foraging_behavior(self, i):
        self.positions[i] += np.random.rand(self.n_features)

    def staying_behavior(self, i):
        self.positions[i] += self.best_positions[i] * np.random.rand(self.n_features)

    def communicating_behavior(self, i):
        self.positions[i] += (self.best_positions.mean(axis=0) - self.positions[i]) * np.random.rand(self.n_features)

    def fear_of_strangers_behavior(self, i):
        self.positions[i] -= np.random.rand(self.n_features)

    def optimize(self, X_train, y_train):
        for t in range(self.max_iter):
            for i in range(self.n_parrots):
                # Evaluate fitness
                fitness = self.fitness(self.positions[i], X_train, y_train)
                
                # Update best positions and fitness
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_positions[i] = self.positions[i]

                # Select behavior randomly
                behavior = np.random.randint(1, 5)
                if behavior == 1:
                    self.foraging_behavior(i)
                elif behavior == 2:
                    self.staying_behavior(i)
                elif behavior == 3:
                    self.communicating_behavior(i)
                elif behavior == 4:
                    self.fear_of_strangers_behavior(i)

        return self.best_positions[np.argmin(self.best_fitness)]


def PO(video,encoded_data):
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    X_train, _, y_train, _ = train_test_split(video, encoded_data, test_size=0.2, random_state=42)
    
    n_features = video.shape[1]
    n_parrots = 20
    max_iter = 10
    
    po = ParrotOptimizer(n_features, n_parrots, max_iter)
    best_features = po.optimize(X_train, y_train)
    # print("Best selected features:", best_features)
    
    return best_features




