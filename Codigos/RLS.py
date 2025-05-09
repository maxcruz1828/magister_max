import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from matplotlib import cm
import pandas as pd

class RLS:
    def __init__(self, order, lam=0.99, delta=1):
        self.order = order
        self.lam = lam
        self.delta = delta

        self.w = np.zeros((order, 3))          # ← predice lat, lon, yaw
        self.P = np.eye(order) / delta         # ← matriz de ganancia

    def predict(self, x):
        x = x.reshape(1, -1)                   
        return x @ self.w                    

    def update(self, x, y):
        x = x.reshape(-1, 1)                   
        y = y.reshape(1, -1)                 

        K = self.P @ x / (self.lam + x.T @ self.P @ x)
        self.w += K @ (y - x.T @ self.w)
        self.P = (self.P - K @ x.T @ self.P) / self.lam
    
def rls_predict_and_train(model, x, cov=None, steps=5):
    preds = []
    current_x = x.copy()  

    for _ in range(steps):
        y_pred = model.predict(current_x)  
        preds.append(y_pred)

        model.update(current_x, y_pred)

        # actualizar ventana de historial
        new_state = y_pred.flatten()     
        current_x = np.roll(current_x, -3)
        current_x[0, -3:] = new_state

    return np.array(preds)

'''

def generate_force_samples(n_samples=1000, mu_fx=0, mu_fy=0, sigma_fx=1, sigma_fy=1, rho=0.5):
    mean = [mu_fx, mu_fy]
    cov = [
        [sigma_fx**2, rho * sigma_fx * sigma_fy],
        [rho * sigma_fx * sigma_fy, sigma_fy**2]
    ]

    samples = np.random.multivariate_normal(mean, cov, n_samples)
    Fx_samples, Fy_samples = samples[:, 0], samples[:, 1]

    return Fx_samples, Fy_samples

def covariance(Fx_samples, Fy_samples, xdot, ydot, m=1.0, Ts=0.1):
    n = len(Fx_samples)
    data = np.zeros((n, 3))  # columnas: x, y, theta

    for i in range(n):
        Fx = Fx_samples[i]
        Fy = Fy_samples[i]

        # Dinámica de un paso
        x = xdot * Ts + (Fx / m) * Ts**2
        y = ydot * Ts + (Fy / m) * Ts**2
        theta = np.arctan(y/x)

        data[i] = [x, y, theta]

    return np.cov(data.T)

'''

# Inicio del programa

def load_trajectory_data(filepath):
    
    # Load the data using pandas
    data = pd.read_csv(filepath)
    
    return data

traj = load_trajectory_data("trayectorias_data/DroneFlightData/WithoutTakeoff/2020-0612/01/0612_2020_113939_01.csv")

traj['time_delta'] = pd.to_timedelta(traj['time'])
traj['segundos'] = traj['time_delta'].dt.total_seconds()
traj['segundos'] = traj['segundos'] - traj['segundos'].min()

# Extract the relevant columns
Traj = traj[['segundos', 'x_gyro', 'y_gyro', 'yaw', 'lat', 'lon', 'alt']]

Traj = Traj.dropna()

# Obtener valores base (primer punto)
lat0 = Traj.iloc[0]['lat']
lon0 = Traj.iloc[0]['lon']
yaw0 = Traj.iloc[0]['yaw']

# Funciones para normalizar y desnormalizar
def normalize_state(lat, lon, yaw):
    return [lat - lat0, lon - lon0, yaw - yaw0]

def denormalize_state(lat_n, lon_n, yaw_n):
    return [lat_n + lat0, lon_n + lon0, yaw_n + yaw0]

# Time
T = len(Traj)
view = 4  # Tamaño de la ventana de entrenamiento

all_preds = []

# Bucle desde view + 1 (para tener suficiente historial real)
for t in range(view + 1, T - 1):
    
    model = RLS(order=3*view)

    # Entrenar con múltiples ejemplos reales antes del paso t
    for j in range(t - view, t):
        history = [
            normalize_state(
                Traj.iloc[i]['lat'],
                Traj.iloc[i]['lon'],
                Traj.iloc[i]['yaw']
            )
            for i in range(j - view, j)
        ]
        Xj = np.array(history).flatten().reshape(1, -1)

        Yj = np.array(normalize_state(
            Traj.iloc[j]['lat'],
            Traj.iloc[j]['lon'],
            Traj.iloc[j]['yaw']
        )).reshape(1, -1)

        model.update(Xj, Yj)

    # Predecir 5 pasos futuros de forma autoregresiva
    x_input = np.array([
        normalize_state(
            Traj.iloc[i]['lat'],
            Traj.iloc[i]['lon'],
            Traj.iloc[i]['yaw']
        )
        for i in range(t - view, t)
    ]).flatten().reshape(1, -1)

    preds = rls_predict_and_train(model, x_input, steps=3)

    # Guardar predicciones
    preds_denorm = np.array([
        denormalize_state(*p.flatten()) for p in preds
    ])
    all_preds.append(preds_denorm)

# Preparación de la gráfica

# Limitar el gráfico a los valores reales
all_lat = Traj['lat'].values
all_lon = Traj['lon'].values

lat_min = min(all_lat)
lat_max = max(all_lat)
lon_min = min(all_lon)
lon_max = max(all_lon)

from matplotlib.animation import FuncAnimation

# Trayectoria real completa
traj_real = Traj[['lat', 'lon']].to_numpy()

# Predicciones (all_preds ya contiene listas de 5 predicciones por paso)
all_preds_array = [np.array(p) for p in all_preds]

# Límites para el gráfico
lat_min, lat_max = traj_real[:, 0].min(), traj_real[:, 0].max()
lon_min, lon_max = traj_real[:, 1].min(), traj_real[:, 1].max()

# Preparar gráfica
fig, ax = plt.subplots(figsize=(8, 6))
real_line, = ax.plot([], [], 'k--', label='Trayectoria real', linewidth=1)
pred_line, = ax.plot([], [], 'ro-', label='Predicción RLS', linewidth=2)

ax.set_xlim(lon_min - 0.0001, lon_max + 0.0001)
ax.set_ylim(lat_min - 0.0001, lat_max + 0.0001)
ax.set_xlabel("Longitud")
ax.set_ylabel("Latitud")
ax.legend()
ax.set_title("Trayectoria real vs 5 predicciones actuales")

# Función de actualización
def actualizacion(frame):
    # Mostrar trayectoria real hasta el frame actual
    real_line.set_data(traj_real[:frame + 1, 1], traj_real[:frame + 1, 0])  # lon, lat

    # Mostrar solo las 5 predicciones hechas en ese frame
    preds = all_preds_array[frame]
    pred_line.set_data(preds[:, 1], preds[:, 0])  # lon, lat

    return real_line, pred_line

ani = FuncAnimation(fig, actualizacion, frames=len(all_preds_array), interval=400, blit=False)
plt.show()


