import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import torch


class LogisticRegressionClassifier:
    def __init__(self, train_features_path, train_labels_path):
        # Load training features and labels
        self.train_features = np.load(train_features_path)
        self.train_labels = np.load(train_labels_path)
        # Initialize candidate parameters for learning rate and regularization coefficient lambda
        self.candidate_learning_rates = [10**(-i) for i in range(1, 6)]
        self.candidate_lambdas = [0] + [10**(-i) for i in range(2,14,2)]
        # Initialize dictionary to hold parameters for one-vs-all classifier
        self.classifiers = {}
        self._initialize_classifiers()


    def _initialize_classifiers(self):
        for label in np.unique(self.train_labels):
            self.classifiers[label] = {
                'weights': None,
                'lambda': None,
                'learning_rate': None
            }
    
    def calculate_probabilities(self, w, X):
        z = np.dot(X, w)
        output = 1 / (1 + np.exp(-z))

        return output
    
    def prediction_function(self, w, X):
        output = self.calculate_probabilities(w, X)
        predictions = np.where(output >= 0.5, 1, -1)

        return predictions

    def fit_one_vs_all_classifier(self, target_class):
        best_accuracy = 0
        best_weights = None
        best_lambda = None
        best_learning_rate = None
        

        # Split training data into n folds (n = number of learning rate and lambda combinations)
        train_features = self.train_features
        train_labels = np.where(self.train_labels == target_class, 1, -1)
        n_splits = len(self.candidate_learning_rates) * len(self.candidate_lambdas)
        splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=544).split(train_features, train_labels))

        # For each candidate learning rate and lambda combination
        for i in range(len(self.candidate_learning_rates)):
            for j in range(len(self.candidate_lambdas)):
                current_learning_rate = self.candidate_learning_rates[i]
                current_lambda = self.candidate_lambdas[j]


                # Use one fold for validation, and other folds for training
                train_indices, validation_indices = splits[i * len(self.candidate_lambdas) + j]
                # Create X_train and y_train folds as pytorch tensors to calculate gradients automatically using autograd functionality of pytorch
                X_train = torch.from_numpy(train_features[train_indices])
                y_train = torch.from_numpy(train_labels[train_indices])
                # Create X_validation and y_validation folds as regular numpy arrays
                X_validation = train_features[validation_indices]
                y_validation = train_labels[validation_indices]
                # X_train, X_validation = train_features[train_indices], train_features[validation_indices]
                # y_train, y_validation = train_labels[train_indices], train_labels[validation_indices]

                # Initialize weights as zeros with gradient tracking
                N, d = X_train.shape
                weights = torch.zeros(d, requires_grad=True, dtype=torch.float64)
                # weights = np.zeros(d)

                max_iterations = 10000
                tolerance = 1e-4

                for _ in range(max_iterations):
                    # Calculate loss in terms of torch tensors
                    term_1 = y_train * torch.matmul(X_train, weights)
                    term_2 = torch.log1p(torch.exp(-term_1))
                    term_3 = torch.sum(term_2) / N
                    regularizer = (current_lambda / 2) * torch.sum(weights ** 2)
                    loss = term_3 + regularizer
                    # term_1 = y_train * np.dot(X_train, weights)
                    # term_2 = np.log(1 + np.exp(term_1))
                    # term_3 = np.sum(term_2) / N
                    # regularizer = (current_lambda / 2) * np.sum(weights ** 2)
                    # loss = term_3 + regularizer

                    # Backpropagate loss to calculate gradients
                    loss.backward()

                    with torch.no_grad():
                        # Save the current weights for convergence check
                        prev_weights = weights.clone().detach()
                        # Get he gradient of loss with respect to weights
                        gradient = weights.grad
                        # Update weights
                        weights -= current_learning_rate * gradient
                        # Zero the gradients for the next iteration
                        weights.grad.zero_()

                    # Check convergence
                    if torch.linalg.norm(weights - prev_weights) < tolerance:
                        break
                
                # Convert weights back to numpy array
                weights = weights.detach().numpy()
                # Calculate accuracy on the validation fold
                y_predicted = self.prediction_function(weights, X_validation)
                accuracy = accuracy_score(y_validation, y_predicted)
                # Update parameters if accuracy improves
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_weights = weights
                    best_lambda = current_lambda
                    best_learning_rate = current_learning_rate
                    

        # Assign the best parameters to the classifier for the target class
        self.classifiers[target_class]['lambda'] = best_lambda
        self.classifiers[target_class]['weights'] = best_weights
        self.classifiers[target_class]['learning_rate'] = best_learning_rate

    def fit(self):
        # Build a one-vs-all classifier for each class
        for target_class in np.unique(self.train_labels):
            self.fit_one_vs_all_classifier(target_class)

    def predict(self, test_features):
        # Initialize empty list to store one-vs-all probabilities for each class
        probabilities = []

        # For each class, calculate the probability of the test features belonging to that class
        for label in sorted(self.classifiers.keys()):
            # Get weights for the classifier of the current class
            weights = self.classifiers[label]['weights']
            # Calculate the probability of the test features belonging to the current class
            class_probabilities = self.calculate_probabilities(weights, test_features)
            # Append to the list of probabilities
            probabilities.append(class_probabilities)

        # Convert the list of probabilities into a numpy array of shape (N_samples, N_classes)
        probabilities = np.stack(probabilities, axis=1)
        # Get the class with the maximum probability for each sample as the predicted label
        y_predicted = np.argmax(probabilities, axis=1)

        return y_predicted




if __name__ == "__main__":
    # Paths to the training data
    train_features_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/train_features.npy"))
    train_labels_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/train_labels.npy"))

    # Create an instance of the LogisticRegressionClassifier
    logistic_regression_classifier = LogisticRegressionClassifier(train_features_path, train_labels_path)
    # Fit the model on the training data
    logistic_regression_classifier.fit()

    # Paths to the test data
    test_features_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/test_features.npy"))
    test_labels_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/test_labels.npy"))
    # Load the test data
    test_features = np.load(test_features_path)
    test_labels = np.load(test_labels_path)

    # Predict the labels for the test set
    y_predicted = logistic_regression_classifier.predict(test_features)

    # Calculate accuracy
    accuracy = accuracy_score(test_labels, y_predicted)

    # Write to the output file
    output_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "logistic_regression_results.txt"))
    with open(output_file_path, "w") as output_file:
        class_names = {0:"rabbit", 1:"yoga", 2:"hand", 3:"snowman", 4:"motorbike"}

        output_file.write(f"Parameters for one-vs-all classifiers:\n\n")
        for label, params in logistic_regression_classifier.classifiers.items():
            output_file.write(f"Classifier-{label}({class_names[label]}):\n")
            output_file.write(f"  Learning Rate: {params['learning_rate']:.4f}\n")
            output_file.write(f"  Lambda: {params['lambda']:.4f}\n")
            output_file.write(f"  Weights: {params['weights']}\n\n")

        output_file.write(f"------------------------------------------------------\n")
        output_file.write(f"Test set accuracy = {accuracy:.4f}\n")