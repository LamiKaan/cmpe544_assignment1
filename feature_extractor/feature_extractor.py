import os
import numpy as np
import mahotas
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle

class FeatureExtractor:
    def __init__(self, train_images_path, train_labels_path, test_images_path, test_labels_path):
        # Load the images and labels of the dataset
        self.train_images = np.load(train_images_path)
        self.train_labels = np.load(train_labels_path)
        self.test_images = np.load(test_images_path)
        self.test_labels = np.load(test_labels_path)
        # Initialize empty lists to store the features
        self.train_features = []
        self.test_features = []
        # Initialize dictionary to store features by class
        self.class_features = self._initalize_class_features()

    def _initalize_class_features(self):
        # Initialize a dictionary to store features by class
        class_features = {}
        for label in np.unique(self.train_labels):
            class_features[label] = []
        return class_features

    def extract_features(self):
        # Extract features from the training images
        for (image, label) in zip(self.train_images, self.train_labels):
            # I used a method called Zernike moments to extract features from the images, which works with binary images. However, the selected threshold to convert a grayscale image to a binary image has a significant impact on the final shape of the object in the binary image, and therefore also on the resulting feature vector. I tried a number of different threshold values and observed how they affect the look of the shapes in the binary image, and finally decided to use a weighted average of features obtained from different thresholds to construct the final feature vector. The weights and thresholds were chosen based on the visual inspection of the resulting binary images.
            binary_image1 = (image > 127).astype(np.uint8)
            binary_image2 = (image > 181).astype(np.uint8)
            binary_image3 = (image > 215).astype(np.uint8)

            feature1 = mahotas.features.zernike_moments(binary_image1, radius=14)
            feature2 = mahotas.features.zernike_moments(binary_image2, radius=14)
            feature3 = mahotas.features.zernike_moments(binary_image3, radius=14)

            feature = 0.2 * feature1 + 0.5 * feature2 + 0.3 * feature3
            # Append the feature vector to the list of features
            self.train_features.append(feature)
            # Also append it to the class specific list
            self.class_features[label].append(feature)

        # Convert the lists of features to numpy arrays
        self.train_features = np.array(self.train_features)
        for label in self.class_features:
            self.class_features[label] = np.array(self.class_features[label])

        # Normalize the features
        scaler = StandardScaler().fit(self.train_features)
        self.train_features = scaler.transform(self.train_features)
        for label in self.class_features:
            self.class_features[label] = scaler.transform(self.class_features[label])

        # Repeat for the test images
        for image in self.test_images:
            binary_image1 = (image > 127).astype(np.uint8)
            binary_image2 = (image > 181).astype(np.uint8)
            binary_image3 = (image > 215).astype(np.uint8)

            feature1 = mahotas.features.zernike_moments(binary_image1, radius=14)
            feature2 = mahotas.features.zernike_moments(binary_image2, radius=14)
            feature3 = mahotas.features.zernike_moments(binary_image3, radius=14)

            feature = 0.2 * feature1 + 0.5 * feature2 + 0.3 * feature3
            self.test_features.append(feature)

        self.test_features = np.array(self.test_features)
        self.test_features = scaler.transform(self.test_features)

    def analyze_intra_class_variations(self):
        class_names = {0:"rabbit", 1:"yoga", 2:"hand", 3:"snowman", 4:"motorbike"}
        output_file_path = os.path.join(os.path.dirname(__file__), "intra_class_variations.txt")
        
        with open(output_file_path, "w") as f:
            
            for label, features in self.class_features.items():
                # Calculate trace and frobenius norm of the covariance matrix for each class
                covariance = np.cov(features, rowvar=False)
                # print(f"Covariance matrix for class {label}:\n{covariance}\n\n")
                trace = np.trace(covariance)
                frobenius_norm = np.linalg.norm(covariance, 'fro')
                # Write the results to the file
                f.write(f"Class {label}({class_names[label]}): Trace = {trace:.4f}, Frobenius Norm = {frobenius_norm:.4f}\n\n")

    def analyze_inter_class_similarities(self):
        class_names = {0: "rabbit", 1: "yoga", 2: "hand", 3: "snowman", 4: "motorbike"}
        output_file_path = os.path.join(os.path.dirname(__file__), "inter_class_similarities.txt")
        
        labels = self.class_features.keys()

        # Calculate mean and covariance matrix for each class
        class_stats = {}
        for label in labels:
            features = self.class_features[label]
            mean = np.mean(features, axis=0)
            covariance = np.cov(features, rowvar=False)
            class_stats[label] = (mean, covariance)

        with open(output_file_path, "w") as f:
            # For every pair of classes; calculate the distance between their means, cosine similarity of their means, and symmetric KL divergence between their distributions as a measure of inter-class similarity
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    mean_i, covariance_i = class_stats[i]
                    mean_j, covariance_j = class_stats[j]

                    # Euclidean distance between means
                    distance = np.linalg.norm(mean_i - mean_j)

                    # Cosine similarity between means
                    cosine_similarity = np.dot(mean_i, mean_j) / (np.linalg.norm(mean_i) * np.linalg.norm(mean_j))

                    # Function to calculate KL divergence between two multivariate Gaussian distributions --> D( N(mu1, cov1) || N(mu2, cov2) )
                    def kl_divergence(mu1, cov1, mu2, cov2):
                        term1 = np.log(np.linalg.det(cov2) / np.linalg.det(cov1))
                        k = len(mu1)
                        term2 = np.trace(np.linalg.inv(cov2) @ cov1)
                        term3 = (mu2 - mu1).T @ np.linalg.inv(cov2) @ (mu2 - mu1)

                        return 0.5 * (term1 - k + term2 + term3)

                    # Calculate KL divergences in both directions
                    kl_ij = kl_divergence(mean_i, covariance_i, mean_j, covariance_j)
                    kl_ji = kl_divergence(mean_j, covariance_j, mean_i, covariance_i)
                    # Take the average of the two KL divergences to get symmetric KL divergence
                    symmetric_kl = 0.5 * (kl_ij + kl_ji)

                    # Write results to file
                    f.write(f"Class {i}({class_names[i]}) vs. Class {j}({class_names[j]}):\n")
                    f.write(f"Euclidean Distance Between Means: {distance:.4f}\n")
                    f.write(f"Cosine Similarity Between Means: {cosine_similarity:.4f}\n")
                    f.write(f"Symmetric KL Divergence: {symmetric_kl:.4f}\n")
                    f.write(f"-----------------------------------------------------\n")

    def plot_features(self):
        class_names = {0: "rabbit", 1: "yoga", 2: "hand", 3: "snowman", 4: "motorbike"}
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        
        # Plot feature vectors with class labels to visualize the distribution of features and observe intra-class variations and inter-class similarities
        # Use t-SNE to reduce dimensionality for visualization
        tsne_train_features = TSNE(n_components=2, random_state=544).fit_transform(self.train_features)
        
        for i, label in enumerate(self.class_features.keys()):
            # Indices of the features belonging to the current class
            indices = np.where(self.train_labels == label)[0]
            # Transform the class features using t-SNE
            tsne_class_features = tsne_train_features[indices]
            # Plot the t-SNE features
            plt.scatter(tsne_class_features[:, 0], tsne_class_features[:, 1], label=class_names[label], color=colors[i], s=1, alpha=0.2)

        plt.legend()
        plt.title("t-SNE Visualization of Feature Vectors")
        plt.grid()
        # plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), "features_tsne_plot.png"), dpi=300)
        plt.show(block=False)

    def save_features(self):
        # Save the features to files
        train_features_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/train_features.npy"))
        test_features_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/test_features.npy"))
        class_features_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/class_features.pkl"))

        # Ensure directories exist
        os.makedirs(os.path.dirname(train_features_path), exist_ok=True)

        # Save the features
        np.save(train_features_path, self.train_features)
        np.save(test_features_path, self.test_features)
        with open(class_features_path, "wb") as f:
            pickle.dump(self.class_features, f)



if __name__ == "__main__":
    # Train images path
    train_images_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/train_images.npy"))
    # Train labels path
    train_labels_path =os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/train_labels.npy"))

    # Test images path
    test_images_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/test_images.npy"))
    # Test labels path
    test_labels_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quickdraw_subset_np/test_labels.npy"))

    # Create an instance of the FeatureExtractor class
    feature_extractor = FeatureExtractor(train_images_path, train_labels_path, test_images_path, test_labels_path)
    # Extract features
    feature_extractor.extract_features()
    # Analyze intra-class variations
    feature_extractor.analyze_intra_class_variations()
    # Analyze inter-class similarities
    feature_extractor.analyze_inter_class_similarities()
    # Plot features
    feature_extractor.plot_features()
    # Save features
    feature_extractor.save_features()
