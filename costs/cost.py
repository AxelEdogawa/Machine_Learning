import numpy as np


class CostFunction:
    """
    Base class for all cost functions.

    This class defines the interface that every cost function
    must implement. Any subclass should provide implementations
    for:
        - compute(): to calculate the cost value
        - gradient(): to calculate gradients w.r.t. parameters

    This allows optimizers to work with any cost function
    interchangeably (polymorphism).
    """

    def compute(self, X, y, w, b):
        """
        Compute the cost value.

        Parameters:
            X : ndarray of shape (m, n)
                Input feature matrix with m samples and n features.
            y : ndarray of shape (m,)
                Target values.
            w : ndarray of shape (n,)
                Weight vector.
            b : float
                Bias term.

        Returns:
            float
                Cost value.
        """
        raise NotImplementedError("compute() not implemented")

    def gradient(self, X, y, w, b):
        """
        Compute the gradients of the cost w.r.t. parameters.

        Parameters:
            X : ndarray of shape (m, n)
                Input feature matrix.
            y : ndarray of shape (m,)
                Target values.
            w : ndarray of shape (n,)
                Weight vector.
            b : float
                Bias term.

        Returns:
            dw : ndarray of shape (n,)
                Gradient of the cost w.r.t. weights.
            db : float
                Gradient of the cost w.r.t. bias.
        """
        raise NotImplementedError("gradient() not implemented")


class MeanSquaredError(CostFunction):
    """
    Mean Squared Error (MSE) cost function for linear regression.

    Cost:
        J(w, b) = (1 / (2m)) * Σ (Xw + b - y)^2

    This class provides both the cost computation and its gradients.
    """

    def compute(self, X, y, w, b):
        """
        Compute the Mean Squared Error cost.

        Parameters:
            X : ndarray of shape (m, n)
                Input feature matrix.
            y : ndarray of shape (m,)
                Target values.
            w : ndarray of shape (n,)
                Weight vector.
            b : float
                Bias term.

        Returns:
            float
                Mean Squared Error cost.
        """
        # Ensure inputs are NumPy arrays with correct types and shapes
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        w = np.asarray(w, dtype=float).reshape(-1)

        # Number of training examples
        m = X.shape[0]

        # Prediction error: (Xw + b) - y
        errors = X @ w + b - y

        # Compute cost: (1 / (2m)) * sum(errors^2)
        return (1 / (2 * m)) * np.dot(errors, errors)

    def gradient(self, X, y, w, b):
        """
        Compute gradients of the MSE cost w.r.t. weights and bias.

        Parameters:
            X : ndarray of shape (m, n)
                Input feature matrix.
            y : ndarray of shape (m,)
                Target values.
            w : ndarray of shape (n,)
                Weight vector.
            b : float
                Bias term.

        Returns:
            dw : ndarray of shape (n,)
                Gradient of the cost w.r.t. weights.
            db : float
                Gradient of the cost w.r.t. bias.
        """
        # Ensure inputs are NumPy arrays with correct types and shapes
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        w = np.asarray(w, dtype=float).reshape(-1)

        # Number of training examples
        m = X.shape[0]

        # Prediction error
        errors = X @ w + b - y

        # Gradient w.r.t. weights: (1 / m) * X^T * errors
        dw = (1 / m) * (X.T @ errors)

        # Gradient w.r.t. bias: (1 / m) * sum(errors)
        db = (1 / m) * np.sum(errors)

        return dw, db