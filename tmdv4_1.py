import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize

# ---- Константы ----
H0 = 69.8          # км/с/Мпк
Omega_m = 0.291
Omega_L = 1.0 - Omega_m
H0_sq = H0**2

# Параметры полей (финальные)
T20 = 0.010
T40 = 1.000
zsym2 = 1000.0
zsym4 = 1.27

# ---- Функция поля (адиабатический трекер) ----
def T_i(z, T0, zsym):
    if z < zsym:
        return T0 * np.sqrt(1 - ((1+z)/(1+zsym))**3)
    else:
        # Экспоненциальное затухание (аппроксимация WKB)
        return T0 * 1e-6 * np.exp(-10 * (z - zsym))

# ---- Вычисление H(z) и w(z) ----
def compute_wz():
    z_grid = np.linspace(0, 3, 300)
    H2 = np.zeros_like(z_grid)
    xi2 = np.zeros_like(z_grid)
    xi4 = np.zeros_like(z_grid)
    for i, z in enumerate(z_grid):
        T2 = T_i(z, T20, zsym2)
        T4 = T_i(z, T40, zsym4)
        xi2[i] = T2 / (1 + T2) if T2 > 0 else 0.0
        xi4[i] = T4 / (1 + T4) if T4 > 0 else 0.0
        H2[i] = H0_sq * (Omega_m * (1+z)**3 * (1 + xi2[i] + xi4[i]) + Omega_L)
    # w(z)
    DeltaH2 = H2 - H0_sq * Omega_m * (1+z_grid)**3
    logDelta = np.log(DeltaH2)
    dlog = np.gradient(logDelta, z_grid)
    w = -1.0 + (1+z_grid)/3.0 * dlog
    return z_grid, w

# ---- Подгонка w(z) под параметризацию w0 + wa*z/(1+z) ----
def fit_w0wa(z, w):
    A = np.column_stack([np.ones_like(z), z/(1+z)])
    w0, wa = np.linalg.lstsq(A, w, rcond=None)[0]
    return w0, wa

# ---- Выполнение и вывод ----
z, w = compute_wz()
w0_fit, wa_fit = fit_w0wa(z, w)
print(f"w0 = {w0_fit:.3f}, wa = {wa_fit:.3f}")

plt.figure(figsize=(10,6))
plt.plot(z, w, 'r-', lw=2, label='ТМДВ‑4.1 (численное решение)')
plt.axhline(-1, color='gray', linestyle=':', label=r'$\Lambda$CDM ($w=-1$)')
plt.xlabel(r'$z$', fontsize=12)
plt.ylabel(r'$w(z)$', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.title('Уравнение состояния тёмной энергии из численного интегрирования')
plt.show()
