import numpy as np
from . import util

from .linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        
        m, n = x.shape
        
        phi = np.sum(y) / m #shape(1,)
        
        mu_0 = np.dot(x.T, 1 - y) / np.sum(1 - y) # shape(n,)
        
        mu_1 = np.dot(x.T, y) / np.sum(y) # shape(n,)
        
        mu_0_r = np.reshape(mu_0, (-1, 1)) # shape(n, 1)
        
        mu_1_r = np.reshape(mu_1, (-1, 1)) # shape(n, 1)
        
        y_r = np.reshape(y, (1, -1)) # shape(1, m)
        
        mu = np.dot(mu_0_r, 1 - y_r) + np.dot(mu_1_r, y_r) #shape(n, m)
        
        sigma = np.dot(x.T - mu, x - mu.T) / m # shape(n, n)
        
        sigma_inv = np.linalg.inv(sigma) # shape(n, n)
        
        theta = np.dot(sigma_inv, mu_1 - mu_0) #shape(n,)
        
        theta_0 = 1 /2 * mu_0 @ sigma_inv @ mu_0 \
                    - 1 / 2 * mu_1 @ sigma_inv @ mu_1 \
                    + np.log((1 - phi) / phi) # shape(1,)
        
        self.theta = np.insert(theta, 0, theta_0)
        
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        
        def hypothesis(theta, x):
            return 1 / (1 + np.exp(- np.dot(x, theta)))
        
        x = util.add_intercept(x)
        
        return hypothesis(self.theta, x) >= 0.5
        # *** END CODE HERE
