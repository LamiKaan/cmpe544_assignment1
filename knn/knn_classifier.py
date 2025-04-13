import os
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


class KNNClassifier:
    def __init__(self, train_features_path, train_labels_path):
        # Load training features and labels
        self.train_features = np.load(train_features_path)
        self.train_labels = np.load(train_labels_path)
        # Initialize candidate parameters for k and distance metrics
        self.candidate_ks = [i for i in range(5, 21, 2)]
        self.candidate_distance_metrics = ["l1", "euclidean"]
        # Initialize variables for best parameters
        self.best_k = None
        self.best_distance_metric = None
        # Initialize KDTree (for fast nearest neighbor search)
        self.kd_tree = None

    def set_kd_tree(self):
        self.kd_tree = KDTree(self.train_features, leaf_size=20, metric=self.best_distance_metric)

    def find_best_parameters(self):
        best_accuracy = 0
        best_k = None
        best_distance_metric = None

        # Split training data into 8 folds (number of candidate k values)
        splits = list(StratifiedKFold(n_splits=len(self.candidate_ks), shuffle=True, random_state=544).split(self.train_features, self.train_labels))

        # Test all candidate k values for each distance metric
        for metric in self.candidate_distance_metrics:
            best_accuracy_for_metric = 0
            best_k_for_metric = None

            # For each candidate k
            for i, k in enumerate(self.candidate_ks):
                # Use the i-th fold for validation, and other 7 folds for training
                train_indices, validation_indices = splits[i]
                X_train, X_validation = self.train_features[train_indices], self.train_features[validation_indices]
                y_train, y_validation = self.train_labels[train_indices], self.train_labels[validation_indices]

                # Create KDTree with the current training folds and the distance metric
                kd_tree = KDTree(X_train, leaf_size=20, metric=metric)
                # Query the KDTree with the validation set and find the k nearest neighbors and their distances for each point in the validation set
                all_distances, all_neighbors = kd_tree.query(X=X_validation, k=k)

                # Initialize an array to store the predicted labels for the current validation fold
                y_predicted = []
                # For each feature in the validation fold
                for distances, neighbor_indices in zip(all_distances, all_neighbors):
                    # Get the labels of its k nearest neighbors
                    neighbor_labels = y_train[neighbor_indices]
                    # Count label occurrences and get the number of maximum occurrence
                    label_counts = np.bincount(neighbor_labels)
                    max_count = np.max(label_counts)
                    # Find label(s) with maximum occurrence
                    max_labels = np.flatnonzero(label_counts == max_count)
                    # If there is only one label with maximum occurrence, then it is the predicted label
                    if len(max_labels) == 1:
                        predicted_label = max_labels[0]
                    # If there is a tie (multiple labels with maximum occurrence), use the nearest neighbor as the tiebreaker
                    else:
                        # Get the distances of the neighbors with maximum occurrening labels
                        max_distances = distances[np.isin(neighbor_labels, max_labels)]
                        # Find the smallest distance among the maximum occurring labels
                        min_distance = np.min(max_distances)
                        # Find the index in distances that corresponds to the smallest distance
                        nearest_index = np.where(distances == min_distance)[0][0]
                        # Get the label of the nearest neighbor
                        predicted_label = neighbor_labels[nearest_index]

                    # Append the predicted label to the list
                    y_predicted.append(predicted_label)

                # Calulate accuracy and update best k for the metric if accuracy improves
                accuracy = accuracy_score(y_validation, y_predicted)
                if accuracy > best_accuracy_for_metric:
                    best_accuracy_for_metric = accuracy
                    best_k_for_metric = k

            # Update overall best k and distance metric if accuracy improves
            if best_accuracy_for_metric > best_accuracy:
                best_accuracy = best_accuracy_for_metric
                best_k = best_k_for_metric
                best_distance_metric = metric

        # Assign the best parameters to the class attributes
        self.best_k = best_k
        self.best_distance_metric = best_distance_metric
        # Set the overall (with all train features) KDTree with the best parameters
        self.set_kd_tree()
        # print(f"Best k: {self.best_k}, Best distance metric: {self.best_distance_metric}")

    def predict(self, test_features):
        # Query the KDTree with the test features and find the k nearest neighbors and their distances
        all_distances, all_neighbors = self.kd_tree.query(X=test_features, k=self.best_k)

        # Initialize an array to store the predicted labels for the test set
        y_predicted = []
        # For each feature in the test set
        for distances, neighbor_indices in zip(all_distances, all_neighbors):
            # Get the labels of its k nearest neighbors
            neighbor_labels = self.train_labels[neighbor_indices]
            # Count label occurrences and get the number of maximum occurrence
            label_counts = np.bincount(neighbor_labels)
            max_count = np.max(label_counts)
            # Find label(s) with maximum occurrence
            max_labels = np.flatnonzero(label_counts == max_count)
            # If there is only one label with maximum occurrence, then it is the predicted label
            if len(max_labels) == 1:
                predicted_label = max_labels[0]
            # If there is a tie (multiple labels with maximum occurrence), use the nearest neighbor as the tiebreaker
            else:
                # Get the distances of the neighbors with maximum occurrening labels
                max_distances = distances[np.isin(neighbor_labels, max_labels)]
                # Find the smallest distance among the maximum occurring labels
                min_distance = np.min(max_distances)
                # Find the index in distances that corresponds to the smallest distance
                nearest_index = np.where(distances == min_distance)[0][0]
                # Get the label of the nearest neighbor
                predicted_label = neighbor_labels[nearest_index]

            # Append the predicted label to the list
            y_predicted.append(predicted_label)

        return np.array(y_predicted).astype(np.int64)


if __name__ == "__main__":
    # Paths to the training data
    train_features_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/train_features.npy"))
    train_labels_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/train_labels.npy"))

    # Create an instance of the KNNClassifier
    knn_classifier = KNNClassifier(train_features_path, train_labels_path)
    # Find the best parameters (k and distance metric)
    knn_classifier.find_best_parameters()

    # Paths to the test data
    test_features_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/test_features.npy"))
    test_labels_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/test_labels.npy"))
    # Load test features and labels
    test_features = np.load(test_features_path)
    test_labels = np.load(test_labels_path)

    # Predict the labels for the test set
    y_predicted = knn_classifier.predict(test_features)

    # Calculate accuracy
    accuracy = accuracy_score(test_labels, y_predicted)

    # Write to the output file
    output_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "knn_classifier_results.txt"))
    with open(output_file_path, "w") as f:
        f.write(f"Selected (best) parameters: Distance Metric =  {knn_classifier.best_distance_metric}, k = {knn_classifier.best_k}\n")
        f.write(f"Test set accuracy = {accuracy:.4f}\n")