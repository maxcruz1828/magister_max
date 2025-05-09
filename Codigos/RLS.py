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
        if cov is not None:
            # Generar ruido gaussiano multivariado
            y_pred += np.random.multivariate_normal([0,0,0], cov)

        preds.append(y_pred)

        model.update(current_x, y_pred)

        # actualizar ventana de historial
        new_state = y_pred.flatten()     
        current_x = np.roll(current_x, -3)
        current_x[0, -3:] = new_state

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

# Inicio del programa

def load_trajectory_data(filepath):
    
    # Load the data using pandas
    data = pd.read_csv(filepath)
    
    return data

traj = load_trajectory_data("trayectorias_data/DroneFlightData/WithoutTakeoff/2020-0612/01/0612_2020_113939_01.csv")
traj['time_delta'] = pd.to_timedelta(traj['time'])
traj['segundos'] = traj['time_delta'].dt.total_seconds()
traj['segundos'] = traj['segundos'] - traj['segundos'].min()
Traj = traj[['segundos', 'x_gyro', 'y_gyro', 'yaw', 'lat', 'lon', 'alt']].dropna()

lat0, lon0, yaw0 = Traj.iloc[0][['lat', 'lon', 'yaw']]
lat_scale = 111_320  # metros por grado latitud
lon_scale = 111_320 * np.cos(np.radians(lat0))  # metros por grado longitud

# Funciones para normalizar y desnormalizar
def normalize_state(lat, lon, yaw):
    lat_m = (lat - lat0) * lat_scale
    lon_m = (lon - lon0) * lon_scale
    yaw_norm = (yaw - yaw0) / 180.0
    return [lat_m, lon_m, yaw_norm]

def denormalize_state(lat_m, lon_m, yaw_norm):
    lat = lat0 + lat_m / lat_scale
    lon = lon0 + lon_m / lon_scale
    yaw = yaw0 + yaw_norm * 180.0
    return [lat, lon, yaw]

# Time
T = len(Traj)
view = 5  # Tamaño de la ventana de entrenamiento

all_preds = []

# Bucle desde view + 1 (para tener suficiente historial real)
for t in range(view + 1, T - 1):
    
    model = RLS(order=3*view)

    # Entrenar con múltiples ejemplos reales antes del paso t
    for j in range(t - view, t):
        history = [
            normalize_state(Traj.iloc[i]['lat'], Traj.iloc[i]['lon'], Traj.iloc[i]['yaw'])
            for i in range(j - view, j)
        ]
        Xj = np.array(history).flatten().reshape(1, -1)
        Yj = np.array(normalize_state(
            Traj.iloc[j]['lat'], Traj.iloc[j]['lon'], Traj.iloc[j]['yaw']
        )).reshape(1, -1)
        model.update(Xj, Yj)

    # Predecir 5 pasos futuros de forma autoregresiva
    x_input = np.array([
        normalize_state(Traj.iloc[i]['lat'], Traj.iloc[i]['lon'], Traj.iloc[i]['yaw'])
        for i in range(t - view, t)
    ]).flatten().reshape(1, -1)

    '''

    # Generar muestras de fuerza
    Fx_samples, Fy_samples = generate_force_samples(n_samples=1000, mu_fx=30, mu_fy=30, sigma_fx=0.1, sigma_fy=0.1, rho=0.5)

    # Calcular la covarianza
    Ts = Traj.iloc[t]['segundos'] - Traj.iloc[t - 1]['segundos']

    xdot = (Traj.iloc[t]['lat'] - Traj.iloc[t - 1]['lat'])/Ts
    ydot = Traj.iloc[t]['lon'] - Traj.iloc[t - 1]['lon']/Ts

    cov = covariance(Fx_samples, Fy_samples, xdot, ydot, m = 3, Ts = Ts)
    '''

    preds = rls_predict_and_train(model, x_input, cov = None, steps=3)

    # Guardar predicciones
    preds_denorm = np.array([denormalize_state(*p.flatten()) for p in preds])
    all_preds.append(preds_denorm)

# Preparación de la gráfica

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

def actualizacion(frame):
    real_index = frame + view + 1
    real_line.set_data(traj_real[:real_index, 1], traj_real[:real_index, 0])
    preds = all_preds_array[frame]
    pred_line.set_data(preds[:, 1], preds[:, 0])
    return real_line, pred_line

ani = FuncAnimation(fig, actualizacion, frames=len(all_preds_array), interval=400, blit=False)
plt.show()


