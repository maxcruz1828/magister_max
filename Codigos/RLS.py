import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from matplotlib import cm


class RLS:
    def __init__(self, order, noise, lam=0.99, delta=1e-2):
        self.order = order
        self.lam = lam
        self.delta = delta
        self.noise = noise

        self.w = np.zeros((order, 1))  # weights
        self.P = np.eye(order) / delta  # inverse correlation matrix
    
    def update(self, x, y):
        # x is the input vector, y is the desired output
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        K = self.P @ x / (self.lam + x.T @ self.P @ x)  # Kalman gain
        self.w = self.w + K @ (y - x.T @ self.w)  # update weights
        self.P = (self.P - K @ x.T @ self.P) / self.lam  # update inverse correlation matrix

    def predict(self, x):
        # x is the input vector
        #x = x.reshape(-1, 1)
        return x.T @ self.w
    
def rls_predict_and_train(model, x, steps=5, cov = None):
    preds = []
    current_x = x.copy()
    for _ in range(steps): 
        pred = model.predict(current_x)

        if cov is not None:
            pred += np.random.multivariate_normal(mean=np.zeros(pred.shape[1]), cov=cov).reshape(pred.shape)

        preds.append(pred)

        model.update(current_x, pred)  # Update model with the prediction as the new target

        current_x = np.roll(current_x, -1)  # Shift the input for the next prediction

    return np.array(preds)

def generate_force_samples(n_samples=1000, mu_fx=0, mu_fy=0, sigma_fx=1, sigma_fy=1, rho=0.5):
    mean = [mu_fx, mu_fy]
    cov = [
        [sigma_fx**2, rho * sigma_fx * sigma_fy],
        [rho * sigma_fx * sigma_fy, sigma_fy**2]
    ]

    samples = np.random.multivariate_normal(mean, cov, n_samples)
    Fx_samples, Fy_samples = samples[:, 0], samples[:, 1]

    return Fx_samples, Fy_samples

def covariance(Fx_samples, Fy_samples, xdot, ydot, m=1.0, Ts=0.1, theta_std=0.05):
    n = len(Fx_samples)
    data = np.zeros((n, 3))  # columnas: x, y, theta

    for i in range(n):
        Fx = Fx_samples[i]
        Fy = Fy_samples[i]

        # Dinámica de un paso
        x = xdot * Ts + (Fx / m) * Ts**2
        y = ydot * Ts + (Fy / m) * Ts**2
        theta = np.random.normal(0, theta_std) * Ts

        data[i] = [x, y, theta]

    return np.cov(data.T)
            
# Inicio de datos

import pandas as pd

def load_trajectory_data(filepath):
    """
    Loads trajectory data from a text file, computes theta, and returns a dataset with x, y, and theta.

    Parameters:
        filepath (str): Path to the trajectory data file.

    Returns:
        np.ndarray: Dataset with columns [x, y, theta].
    """
    
    # Load the data using pandas
    data = pd.read_csv(filepath, sep=r"\s+", skiprows=1, names=["X", "Y", "Z"])

    # Compute theta as the angle of movement (arctan of y/x)
    data["theta"] = np.arctan2(data["Y"].diff(), data["X"].diff()).fillna(0)

    # Return only X, Y, and theta as a NumPy array
    return data




