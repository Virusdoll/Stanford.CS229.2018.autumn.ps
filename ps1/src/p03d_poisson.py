import numpy as np
from . import util

from .linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def hypothesis(self, theta, x):
        # (m, n) @ (n,) -> (m,)
        return np.exp(x @ theta) 
    
    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***

        def step(theta, x, y, alpha):
            # (m,) @ (m, n) -> (n,)
            return alpha * (y - self.hypothesis(theta, x)) @ x
        
        # init theta
        m, n = x.shape
        self.theta = np.zeros(n)
        
        while True:
            new_step = step(self.theta, x, y, self.step_size)
            if np.linalg.norm(new_step, ord=1) < self.eps: break
            self.theta += new_step
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        
        return self.hypothesis(self.theta, x)
        
        # *** END CODE HERE ***
