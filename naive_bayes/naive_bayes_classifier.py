import os
import numpy as np
from sklearn.metrics import accuracy_score


class NaiveBayesClassifier:
    def __init__(self, train_features_path, train_labels_path):
        # Load training features and labels
        self.train_features = np.load(train_features_path)
        self.train_labels = np.load(train_labels_path)
        self.class_priors = self._calculate_class_priors()
        self.class_means = self._calculate_class_means()
        self.class_variances = self._calculate_class_variances()

    def _calculate_class_priors(self):
        # Initialize empty dictionary to store priors for each class
        priors = {}

        # Get unique class labels and their counts
        class_labels, counts = np.unique(self.train_labels, return_counts=True)
        # Get the total number of training samples
        total_count = len(self.train_labels)

        # Calculate the prior probability for each class
        for label, count in zip(class_labels, counts):
            # Calculate prior as the count of samples in the class divided by total samples
            priors[label] = count / total_count

        return priors
    
    def _calculate_class_means(self):
        # Initialize empty dictionary to store means for each class
        means = {}

        # For each class
        for label in np.unique(self.train_labels):
            # Get the samples belonging to the current class
            class_samples = self.train_features[self.train_labels == label]
            # Calculate the mean of the samples in the class
            means[label] = np.mean(class_samples, axis=0)

        return means
    
    def _calculate_class_variances(self):
        # Initialize empty dictionary to store variances for each class
        variances = {}

        # For each class
        for label in np.unique(self.train_labels):
            # Get the samples belonging to the current class
            class_samples = self.train_features[self.train_labels == label]
            # Calculate the variance of the samples in the class
            variances[label] = np.var(class_samples, axis=0)

        return variances
    
    def predict(self, test_features):
        # Initialize an array to store the predicted labels
        y_predicted = []

        # For each test sample
        for sample in test_features:
            # Initialize a dictionary to store the log-posterior (for numerical stability in case of very small float values) probabilities for each class
            log_posteriors = {}

            # Calculate the log-posterior probability of each class given the test sample
            for label in self.class_priors.keys():
                # Get log-prior for the class
                log_prior = np.log(self.class_priors[label])
                # Get the mean and variances for the class
                mean = self.class_means[label]
                variances = self.class_variances[label]

                # Calculate the log-likelihood using the Gaussian distribution formula
                term_1 = np.log(2 * np.pi * variances)
                term_2 = ((sample - mean) ** 2) / (2 * variances)
                log_likelihood = -0.5 * np.sum(term_1 + term_2)
                
                # Calculate the log-posterior probability (offset with a constant by omitting log-evidence which is constant for all classes)
                log_posteriors[label] = log_prior + log_likelihood

            # Choose the class with the highest posterior probability
            predicted_label = max(log_posteriors, key=log_posteriors.get)
            y_predicted.append(predicted_label)

        return np.array(y_predicted).astype(np.int64)

if __name__ == "__main__":
    # Paths to the training data
    train_features_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/train_features.npy"))
    train_labels_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/train_labels.npy"))

    # Create an instance of the NaiveBayesClassifier
    naive_bayes_classifier = NaiveBayesClassifier(train_features_path, train_labels_path)

    # Paths to the test data
    test_features_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/test_features.npy"))
    test_labels_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/test_labels.npy"))
    # Load test features and labels
    test_features = np.load(test_features_path)
    test_labels = np.load(test_labels_path)

    # Predict the labels for the test set
    y_predicted = naive_bayes_classifier.predict(test_features)

    # Calculate accuracy
    accuracy = accuracy_score(test_labels, y_predicted)

    # Write to the output file
    output_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "naive_bayes_classifier_results.txt"))
    with open(output_file_path, "w") as f:
        f.write(f"Test set accuracy = {accuracy:.4f}\n")

    