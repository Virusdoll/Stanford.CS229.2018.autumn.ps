import numpy as np
from . import util

from .linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        
        def hypothesis(theta, x):
            '''
            n, m*n -> m
            '''
            return 1 / (1 + np.exp(- np.dot(x, theta)))
        
        def gradient(theta, x, y):
            '''
            n, m*n, m -> n
            '''
            m, n = x.shape
            return - 1 / m * np.dot(x.T, y - hypothesis(theta, x))
        
        def hessian(theta, x):
            '''
            n, m*n -> n*n
            '''
            m, n = x.shape
            h = np.reshape(hypothesis(theta, x), (-1, 1))
            return 1 / m * np.dot(x.T, h * (1 - h) * x)
        
        def newton_method(theta, x, y):
            '''
            n, m*n, m -> n
            '''
            return theta - np.dot(
                np.linalg.inv(hessian(theta, x)),
                gradient(theta, x, y)
            )
        
        epsilon = 0.00005
        m, n = x.shape
        
        if self.theta is None:
            self.theta = np.zeros(n)
        
        while True:
            old_theta = self.theta
            self.theta = newton_method(old_theta, x, y)
            
            if np.linalg.norm(self.theta - old_theta, ord=1) < epsilon:
                break
        
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
            '''
            as same as hypothesis in fit
            '''
            return 1 / (1 + np.exp(- np.dot(x, theta)))
        
        return hypothesis(self.theta, x) >= 0.5
        
        # *** END CODE HERE ***
