import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import matplotlib.patches as patches


class ExpectationMaximization:
    def __init__(self, dataset_path):
        # Load the dataset
        self.dataset = np.load(dataset_path)
        # Get number of data points and their dimensions
        self.N, self.d = self.dataset.shape
        # Set number of clusters (data is generated from 3 2-D Gaussian distributions)
        self.K = 3
        # Initialize means and covariance matrices (approximated based on the visual inspection of the scatter plot)
        self.means = np.array([[0, 1], [4, 5], [9, 9]]).astype(float)
        self.covariances = np.array([[[1.5, 0], [0, 1]], [[1.5, 0], [0, 1]], [[1.5, -1], [-1, 1]]]).astype(float)
        # Initialize mixing coefficients as equal
        self.pi = np.ones(self.K).astype(float) / self.K
        # Initialize responsibilities as an empty matrix
        self.gamma = np.zeros((self.N, self.K)).astype(float)
        # Initialize log-likelihood as zero
        self.log_likelihood = 0.0

    def create_scatter_plot(self):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.dataset[:, 0], self.dataset[:, 1], s=30, alpha=0.7)
        plt.title("Scatter Plot of Dataset")
        plt.xlabel("x₁")
        plt.ylabel("x₂")
        plt.grid(True)
        plt.savefig(os.path.join(os.path.dirname(__file__), "em_scatter_plot.png"), dpi=300)
        plt.show(block=False)

    def expectation_step(self):
        # Calculate the responsibilities (gamma)
        for k in range(self.K):
            self.gamma[:, k] = self.pi[k] * multivariate_normal.pdf(self.dataset, mean=self.means[k], cov=self.covariances[k])

        # Normalize the responsibilities
        self.gamma /= np.sum(self.gamma, axis=1, keepdims=True)

    def maximization_step(self):
        # Calculate the effective number of points assigned to each cluster
        N_k = np.sum(self.gamma, axis=0)
        # Update the mixing coefficients
        self.pi = N_k / self.N
        # Calulate the new means
        self.means = (self.gamma.T @ self.dataset) / N_k[:, np.newaxis]
        # Calculate the new covariance matrices using the new means
        for k in range(self.K):
            diff = self.dataset - self.means[k]
            r_k = self.gamma[:, k].reshape(-1, 1)
            self.covariances[k] = (r_k * diff).T @ diff / N_k[k]
            # print(self.covariances[k])

    def compute_log_likelihood(self):
        # Initialize an empty likelihood matrix
        likelihoods = np.zeros((self.N, self.K))

        # Compute the weighted likelihoods for all points and components
        for k in range(self.K):
            likelihoods[:, k] = self.pi[k] * multivariate_normal.pdf(self.dataset, mean=self.means[k], cov=self.covariances[k])

        # Sum over components and take the log, then sum over all data points to get the total log-likelihood
        new_log_likelihood = np.sum(np.log(np.sum(likelihoods, axis=1)))
        return new_log_likelihood
    
    def fit(self, max_iter=1000, tol=1e-4):
        # Apply expectation and maximization steps in sequence until the difference between the new and previous log-likelihood is smaller than 0.01%, or until the maximum number of iterations
        for i in range(max_iter):
            # print(f"Iteration {i+1}/{max_iter}")
            # E-step
            self.expectation_step()

            # M-step
            self.maximization_step()

            # Compute new log-likelihood
            new_log_likelihood = self.compute_log_likelihood()

            if abs(new_log_likelihood - self.log_likelihood) / abs(new_log_likelihood) < tol:
                break
            else:
                self.log_likelihood = new_log_likelihood

    def plot_results(self):
        # Define a list of distinct colors for the clusters
        colors = ['blue', 'orange', 'purple']
        # Assign each point to the cluster with highest responsibility
        cluster_assignments = np.argmax(self.gamma, axis=1)
        
        # Plot data points colored by cluster assignment
        plt.figure(figsize=(8, 6))
        for k in range(self.K):
            cluster_points = self.dataset[cluster_assignments == k]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[k], label=f"Cluster {k+1}", s=30, alpha=0.6)

        plt.title("EM Cluster Assignments")
        plt.xlabel("x₁")
        plt.ylabel("x₂")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(os.path.dirname(__file__), "em_cluster_assignments.png"), dpi=300)
        plt.show(block=False)

        plt.figure(figsize=(8, 6))
        for k in range(self.K):
            cluster_points = self.dataset[cluster_assignments == k]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[k], label=f"Cluster {k+1}", s=30, alpha=0.2)

            # Plot Gaussian ellipse
            mean = self.means[k]
            cov = self.covariances[k]
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            order = eigenvalues.argsort()[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            width, height = 2 * np.sqrt(eigenvalues)
            ellipse = patches.Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor=colors[k], facecolor='none', linewidth=2.0)
            plt.gca().add_patch(ellipse)

            # Add the mean point
            plt.scatter(*mean, color=colors[k], edgecolor='black', marker='X', s=100, zorder=5)

            # Add arrows for principal axes
            for i in range(2):
                vec = eigenvectors[:, i]
                length = np.sqrt(eigenvalues[i])
                plt.arrow(mean[0], mean[1], vec[0]*length, vec[1]*length,
                          color=colors[k], width=0.05, head_width=0.2, alpha=0.8)

        plt.title("Estimated Gaussian Distributions")
        plt.xlabel("x₁")
        plt.ylabel("x₂")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(os.path.dirname(__file__), "em_gaussian_distributions.png"), dpi=300)
        plt.show(block=False)

    def save_results(self):
        # Write results to file
        with open(os.path.join(os.path.dirname(__file__), "em_results.txt"), "w") as f:
            for k in range(self.K):
                f.write(f"Cluster {k+1}:\n")
                f.write(f"Mean:\n{self.means[k]}\n")
                f.write(f"Covariance:\n{self.covariances[k]}\n\n")



if __name__ == "__main__":
    # Get the absolute path of the dataset relative to the current file
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/dataset.npy"))

    # Create an instance of the ExpectationMaximization class
    em = ExpectationMaximization(dataset_path)

    # Run the EM algorithm
    em.create_scatter_plot()
    em.fit()
    em.plot_results()
    em.save_results()
