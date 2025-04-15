import numpy as np

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
        x = x.reshape(-1, 1)
        return x.T @ self.w
    
def rls_predict_and_train(model, x, steps=5):
    preds = []
    current_x = x.copy()
    for _ in range(steps): 
        pred = model.predict(current_x)
        preds.append(pred)

        model.update(current_x, pred)  # Update model with the prediction as the new target

        current_x = np.roll(current_x, -1)  # Shift the input for the next prediction

    return np.array(preds)