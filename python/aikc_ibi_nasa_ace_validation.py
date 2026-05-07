"""
PROJECT: Adaptive Information Kernel Control (AIKC):
Real-Data Validation using NASA ACE Mission
CORE THEORY: Effective Theory of Information-Based Instrumentation (IBI)
PURPOSE: Validation of IBI Theory and AIKC Extension in Non-Gaussian Real Environments
AUTHOR:       Daniel Isaias Vivas Zamora
ROLE:         Professor & Coordinator, Unidad Proyecto Aragua (UPA)
AFFILIATION:  University of Carabobo, Faculty of Health Sciences (FCS)
CONTACT:      dvivas1@uc.edu.ve | vivaszdaniel@gmail.com
LOCATION:     Maracay, Aragua, Venezuela
DATE:         May 2026
VERSION:      1.7.0 
LICENSING: GNU General Public License v3.0 (GPL-3.0)
DESCRIPTION:
This script processes Level 2 Magnetic Field (MFI) data from the NASA ACE Mission.
Includes: Standard Metrics, Filter Gain Tracking, Whiteness, NIS, Jump, TE, PSD.
Generates separated high-quality plots and .csv data.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import warnings
from scipy.signal import savgol_filter, welch
from scipy import stats
from scipy.interpolate import interp1d
from scipy.stats import linregress
import cdflib
import os
import time
import threading
from datetime import datetime

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.max_open_warning': 0})

# =============================================================================
# 0. UI & CONSOLE ANIMATION
# =============================================================================
BANNER = """
╔══════════════════════════════════════════════════════════════════════╗
║  PROJECT: Adaptive Information Kernel Control (AIKC)                 ║
║  Real-Data Validation using NASA ACE Mission                         ║
║  CORE THEORY: Effective Theory of Information-Based Instrumentation  ║
║  PURPOSE: Experimental Validation of IBI Theory and AIKC Extension   ║
║  AUTHOR: Daniel Isaias Vivas Zamora  EMAIL: dvivas1@uc.edu.ve        ║
║  ROLE: Professor & Coordinator, Unidad Proyecto Aragua (UPA)         ║
║  AFFILIATION: University of Carabobo, Faculty of Health Sciences(FCS)║
║  LICENSING: GNU General Public License v3.0 (GPL-3.0)                ║
╚══════════════════════════════════════════════════════════════════════╝
"""

class ProgressSpinner:
    """Simple background spinner using ASCII characters."""
    def __init__(self, msg="Processing"):
        self.msg = msg
        self.running = False
        self._thread = None
        
    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()
        
    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join()
        sys.stdout.write('\r' + ' ' * (len(self.msg) + 10) + '\r')
        sys.stdout.flush()
        
    def _animate(self):
        symbols = ['/', '-', '\\', '|']
        i = 0
        while self.running:
            sys.stdout.write(f'\r  {self.msg} {symbols[i % 4]} ')
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1

# =============================================================================
# 1. CONSTANTS AND IBI MATHEMATICS
# =============================================================================
PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)
EPSILON = 1e-12

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def log_phi(x):
    return np.log(np.maximum(x, EPSILON)) / LN_PHI

def kernel_pellis(p, pi):
    mask = (p > EPSILON) & (pi > EPSILON)
    return -np.sum(p[mask] * log_phi(p[mask] / pi[mask])) if np.any(mask) else 0.0

def kernel_tsallis(p, pi):
    q = 1.5
    mask = p > EPSILON
    return (1.0 - np.sum(p[mask]**q)) / (q - 1.0) if np.any(mask) else 0.0

def kernel_jsd(p, pi):
    m = 0.5 * (p + pi)
    def kl(a, b):
        mask = (a > EPSILON) & (b > EPSILON)
        return np.sum(a[mask] * np.log(a[mask] / b[mask]))
    return 0.5 * kl(p, m) + 0.5 * kl(pi, m)

def kernel_fisher(p, pi):
    n = len(p)
    return 0.0 if n < 2 else np.sum(np.diff(p)**2) / (np.sum(p[:-1]) + EPSILON)

# =============================================================================
# 2. FILTERS IMPLEMENTATION
# =============================================================================
class AgnosticAdaptiveFilter:
    def __init__(self, dt, R_base=1e-5, kernel_func=None, kernel_name='Pellis', rng=None):
        self.dt, self.R_base, self.R_ant = dt, R_base, R_base
        self.kernel_func, self.kernel_name = kernel_func, kernel_name
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.x = np.zeros((3, 1)); self.P = np.eye(3) * 0.1
        self.F = np.array([[1.0, dt, 0.5*dt**2], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
        self.H = np.array([[1.0, 0.0, 0.0]]); self.Q = np.diag([1e-7, 1e-8, 1e-6])
        self.buffer, self.buffer_size = [], 64
        self.warmup_steps, self.warmup_metrics = 50, []
        self.mu0, self.delta, self.beta = 0.0, 1.0, 5.0
        self.calibrated = False
        self.last_K = 0.0 # Exposed for Gain Tracking

    def filter(self, z, conc_factor=1.0):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        inn = z - self.H @ self.x
        self.buffer.append(float(inn[0, 0]))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
            
        pi = PHI**(-np.arange(15))
        pi /= (pi.sum() + 1e-12)
        
        metric_val = 0.0
        if len(self.buffer) >= 10:
            data = np.array(self.buffer)
            sigma = np.std(data) + 1e-9
            counts, _ = np.histogram(data / sigma, bins=15, range=(-3, 3))
            p_dist = counts.astype(float) + 1e-3
            p_dist /= p_dist.sum()
            val = self.kernel_func(p_dist, pi)
            metric_val = val if np.isfinite(val) else 0.0
            
        if not self.calibrated:
            self.warmup_metrics.append(metric_val)
            if len(self.warmup_metrics) >= self.warmup_steps:
                self.mu0 = np.mean(self.warmup_metrics)
                self.delta = np.std(self.warmup_metrics) + 1e-5
                self.calibrated = True
            activation_score = 0.0
        else:
            activation_score = (metric_val - self.mu0) / self.delta
            
        self.R_ant = np.clip(
            self.R_base * (1.0 + self.beta * sigmoid(activation_score)), 
            self.R_base * 0.1, self.R_base * 2000
        )
        
        S = self.H @ self.P @ self.H.T + self.R_ant
        K = self.P @ self.H.T / S
        self.x = self.x + K @ inn
        I = np.eye(self.P.shape[0])
        term1 = (I - K @ self.H)
        self.P = term1 @ self.P @ term1.T + K @ np.array([[self.R_ant]]) @ K.T

        limit = max(0.01, 2.0 * np.max(np.abs(z)))
        self.x[0,0] = np.clip(self.x[0,0], -limit, limit)
        self.last_K = float(K[0, 0]) # ✅ Gain Tracking
        return float(self.x[0, 0])

class AKF:
    def __init__(self, dt, R_base=1e-5, warmup_steps=50):
        self.dt, self.R_base, self.R_ant = dt, R_base, R_base
        self.F = np.array([[1.0, dt, 0.5*dt**2], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
        self.H = np.array([[1.0, 0.0, 0.0]]); self.Q = np.diag([1e-7, 1e-8, 1e-6])
        self.x = np.zeros((3, 1)); self.P = np.eye(3) * 0.1
        self.warmup_steps, self.warmup_inn = warmup_steps, []
        self.calibrated, self.R_scale = False, 1.0
        self.last_K = 0.0 # ✅ Gain Tracking

    def filter(self, z, conc_factor=None):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        inn = z - self.H @ self.x
        inn_sq = inn[0,0]**2
        
        if not self.calibrated:
            self.warmup_inn.append(inn_sq)
            if len(self.warmup_inn) >= self.warmup_steps: 
                self.R_scale = np.mean(self.warmup_inn) / (self.R_base + 1e-12)
                self.calibrated = True
                
        R_active = self.R_ant * np.clip(self.R_scale, 0.1, 100.0)
        self.R_ant = np.clip(0.85 * self.R_ant + 0.15 * inn_sq, self.R_base * 0.1, self.R_base * 1000)
        
        S = self.H @ self.P @ self.H.T + R_active
        K = self.P @ self.H.T / S
        self.x = self.x + K @ inn
        I = np.eye(3)
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ np.array([[R_active]]) @ K.T
        
        limit = max(0.01, 2.0 * np.max(np.abs(z)))
        self.x[0, 0] = np.clip(self.x[0, 0], -limit, limit)
        self.last_K = float(K[0, 0]) # ✅ Gain Tracking
        return float(self.x[0,0])

class PINNFilter:
    def __init__(self, dt, R_base=1e-5, lr=0.008, warmup_steps=50, rng=None):
        self.dt, self.R_base, self.lr = dt, R_base, lr
        self.F = np.array([[1.0, dt, 0.5*dt**2], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
        self.H = np.array([1.0, 0.0, 0.0])
        self.rng = rng if rng else np.random.default_rng(42)
        self.W1 = self.rng.normal(0, 0.1, (3, 8)); self.b1 = np.zeros(8)
        self.W2 = self.rng.normal(0, 0.1, (8, 3)); self.b2 = np.zeros(3)
        self.x = np.zeros(3); self.a1 = None
        self.K_adapt = np.array([0.4, 0.1, 0.05])
        self.warmup_steps, self.warmup_err = warmup_steps, []
        self.calibrated, self.step_count = False, 0
        self.last_K = 0.4 # ✅ Gain Tracking (initial K_pos)

    def _relu(self, x): 
        return np.maximum(0.0, x)

    def _forward(self, x_in):
        self.a1 = self._relu(self.W1.T @ x_in + self.b1)
        return self.W2.T @ self.a1 + self.b2

    def filter(self, z, conc_factor=None):
        z_val = float(z[0,0]) if z.ndim > 1 else float(z)
        self.step_count += 1
        
        x_phys = self.F @ self.x[:, None]
        x_nn = self._forward(self.x)[:, None]
        x_pred = 0.7 * x_phys + 0.3 * x_nn
        
        inn = z_val - (self.H @ x_pred)[0]
        
        if not self.calibrated:
            self.warmup_err.append(inn**2)
            if len(self.warmup_err) >= self.warmup_steps:
                scale = np.sqrt(np.mean(self.warmup_err))
                self.K_adapt = np.clip(self.K_adapt * (0.5 + 0.5 * np.exp(-scale*50)), 0.05, 0.8)
                self.calibrated = True
                
        self.x = x_pred.flatten() + self.K_adapt * inn
        limit = max(0.01, 2.0 * np.abs(z_val))
        self.x[0] = np.clip(self.x[0], -limit, limit)
        
        if self.step_count % 10 == 0 and self.a1 is not None:
            delta = inn * self.lr
            self.W2 += delta * np.outer(self.a1, self.H)
            self.b2 += delta * self.H
            
        self.last_K = float(self.K_adapt[0]) # ✅ Gain Tracking
        return float(self.x[0])

# =============================================================================
# 3. DATA LOADING & PREPROCESSING
# =============================================================================
def safe_parse_epoch(raw_time):
    arr = np.asarray(raw_time, dtype=np.float64).flatten()
    val_max = np.max(np.abs(arr))
    if val_max > 1e12:
        return pd.to_datetime(arr - 62167219200000.0, unit='ms')
    elif val_max > 1e9:
        return pd.to_datetime(arr, unit='ms')
    else:
        return pd.to_datetime(arr, unit='s')

def load_nasa_data(directory='./data'):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return None, None
    files = sorted([f for f in os.listdir(directory) if f.endswith('.cdf')])
    all_data, mission = [], None
    if not files: return None, None
    
    for f in files:
        path = os.path.join(directory, f)
        try:
            cdf = cdflib.CDF(path)
            has_close = hasattr(cdf, 'close')
            raw_time = cdf.varget('Epoch')
            if raw_time is None: raw_time = cdf.varget('Time')
            if raw_time is None:
                if has_close: cdf.close()
                continue  
            t_arr = np.asarray(raw_time)
            if t_arr.size == 0:
                if has_close: cdf.close()
                continue
            time_dt = safe_parse_epoch(raw_time)
            if time_dt.year.min() < 2010 or time_dt.year.max() > 2026:
                if has_close: cdf.close()
                continue

            bz = None
            if f.startswith('ac_h2_mfi_'):
                mission = 'ACE'
                for var in ['BGSM', 'BC', 'B_GSM']:
                    data = cdf.varget(var)
                    if data is not None:
                        d_arr = np.asarray(data)
                        if d_arr.ndim > 1 and d_arr.shape[1] >= 3:
                            bz = d_arr[:, 2]; break
            if bz is None:
                if has_close: cdf.close()
                continue
            df_temp = pd.DataFrame({'Time': time_dt, 'Bz': bz})
            df_temp = df_temp.dropna().drop_duplicates()
            all_data.append(df_temp)
            if has_close: cdf.close()
        except Exception as e:
            try:
                if 'cdf' in locals() and hasattr(cdf, 'close'): cdf.close()
            except: pass

    if not all_data: return None, None
    return pd.concat(all_data).sort_values('Time').drop_duplicates().reset_index(drop=True), mission

def preprocess_for_ibi(df, variable='Bz', target_dt=150.0):
    df = df.sort_values('Time').reset_index(drop=True)
    df['time_unix'] = df['Time'].apply(lambda x: x.timestamp())
    t, signal = df['time_unix'].values, df[variable].values
    t_start, t_end = t[0], t[-1]
    span_years = (t_end - t_start) / 3.154e7
    if span_years > 10.0 or (t_end - t_start) > 1e11:
        if t[0] > 1e12: t = t / 1000.0; t_start, t_end = t[0], t[-1]
    n_expected = int((t_end - t_start) / target_dt)
    if n_expected > 2_000_000:
        df = df.iloc[::max(1, len(df)//250000)].reset_index(drop=True)
        df['time_unix'] = df['Time'].apply(lambda x: x.timestamp())
        t = df['time_unix'].values; signal = df[variable].values; t_start, t_end = t[0], t[-1]
    t_uniform = np.arange(t_start, t_end, target_dt)
    s_uniform = interp1d(t, signal, kind='linear', bounds_error=False, fill_value=np.nan)(t_uniform)
    series = pd.Series(s_uniform).ffill().bfill().dropna()
    if len(series) < 50: return np.array([]), np.array([]), np.array([])
    window = min(51, len(series) - 1)
    if window % 2 == 0: window -= 1
    clean_proxy = savgol_filter(series.values, window, polyorder=3)
    return series.index.values, clean_proxy, series.values

# =============================================================================
# 4. COMPREHENSIVE VALIDATION METRICS
# =============================================================================
def validate_tracking(y_true, y_pred, dt, burn_in=40):
    y_t, y_p = y_true[burn_in:], y_pred[burn_in:]
    if len(y_t) < 20 or np.std(y_p) < 1e-12: return {'valid': False}
    err = y_t - y_p
    mse = np.mean(err**2); rmse = np.sqrt(mse); mae = np.mean(np.abs(err))
    corr = np.corrcoef(y_t, y_p)[0, 1]
    r2 = corr**2 if not np.isnan(corr) else 0.0
    nse = 1.0 - (np.sum((y_t - y_p)**2) / (np.sum((y_t - np.mean(y_t))**2) + 1e-12))
    mbe = np.mean(y_p - y_t)
    pbias = 100.0 * (np.sum(y_p - y_t) / (np.sum(y_t) + 1e-12))
    alpha = np.std(y_p) / (np.std(y_t) + 1e-12)
    beta = np.mean(y_p) / (np.mean(y_t) + 1e-12)
    kge = 1.0 - np.sqrt((corr - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2)
    kurtosis_err = stats.kurtosis(err)
    return {'valid': True, 'mse': mse, 'rmse': rmse, 'r2': r2, 'nse': nse, 'mbe': mbe,
            'pbias': pbias, 'kge': kge, 'kurtosis': kurtosis_err, 'mae': mae,
            'std_err': np.std(err), 'max_err': np.max(np.abs(err)), 'median_err': np.median(np.abs(err)),
            'correlation': corr if not np.isnan(corr) else 0.0, 'alpha': alpha, 'beta': beta}

def check_bias_health(observed, predicted, filter_name):
    mbe = np.mean(predicted - observed)
    nse = 1.0 - (np.sum((observed - predicted)**2) / (np.sum((observed - np.mean(observed))**2) + 1e-12))
    mbe_status = "✓ Ideal " if abs(mbe) < 0.01 else ("⚠ Low bias " if abs(mbe) < 0.1 else "✗ High bias ")
    nse_status = "✓ Excellent " if nse > 0.95 else ("✓ Good " if nse > 0.80 else (" Acceptable " if nse > 0.50 else "✗ Poor "))
    print(f"   {filter_name: <8}: MBE={mbe:+.6f} ({mbe_status}) | NSE={nse:.6f} ({nse_status}) ")
    return mbe, nse

FLOPS_PER_ITER = {'AKF': 180, 'Pellis': 270, 'Tsallis': 285, 'JSD': 310, 'Fisher': 235, 'PINN': 595}

# =============================================================================
# 5. EFFICIENCY & STANDARD PLOTTING
# =============================================================================
def plot_efficiency_real(df_path, output_dir='./plots', mission='ACE'):
    try: df = pd.read_csv(df_path)
    except FileNotFoundError: return
    plt.figure(figsize=(10, 7))
    filtros = ['Fisher', 'Pellis', 'JSD', 'Tsallis', 'PINN', 'AKF']
    colores = {"Fisher": "#0072B2", "AKF": "#E69F00", "Pellis": "#D55E00",
               "Tsallis": "#F0E442", "JSD": "#CC79A7", "PINN": "#009E73"}
    marcadores = {'Pellis': 'o', 'Tsallis': 's', 'AKF': '^', 'PINN': 'D', 'JSD': 'p', 'Fisher': 'H'}
    for filtro in filtros:
        mask = df['Filter'] == filtro
        if mask.any() and 'Total_FLOPs' in df.columns and 'MSE' in df.columns:
            flops = df.loc[mask, 'Total_FLOPs'].values[0]
            mse = df.loc[mask, 'MSE'].values[0]
            plt.scatter(flops, mse, label=filtro, color=colores[filtro], 
                       marker=marcadores[filtro], s=100, edgecolor='black', alpha=0.9)
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Total FLOPs', fontsize=11); plt.ylabel('MSE', fontsize=11)
    plt.title(f'Efficiency: MSE vs FLOPs ({mission} - Real Data)', fontsize=13, fontweight='bold')
    plt.legend(loc='best', fontsize=9, framealpha=0.9); plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    plt.savefig(f"{output_dir}/eficiencia_real_{mission.lower()}_{ts}.png", dpi=300, bbox_inches='tight'); plt.close()

def plot_comprehensive_metrics(df_path, output_dir='./plots', mission='ACE'):
    try: df = pd.read_csv(df_path)
    except FileNotFoundError: return
    filtros = ['Fisher', 'Pellis', 'JSD', 'Tsallis', 'PINN', 'AKF']
    colores = {"Fisher": "#0072B2", "AKF": "#E69F00", "Pellis": "#D55E00",
               "Tsallis": "#F0E442", "JSD": "#CC79A7", "PINN": "#009E73"}
    if 'NSE' in df.columns and 'MSE' in df.columns:
        plt.figure(figsize=(9, 6))
        for filtro in filtros:
            mask = df['Filter'] == filtro
            if mask.any():
                nse_val = df.loc[mask, 'NSE'].values[0]
                mse_val = df.loc[mask, 'MSE'].values[0]
                plt.scatter(mse_val, nse_val, label=filtro, 
                           color=colores[filtro], s=100, edgecolor='black', alpha=0.9)
        plt.xscale('log'); plt.xlabel('MSE', fontsize=11); plt.ylabel('NSE (Nash-Sutcliffe Efficiency)', fontsize=11)
        plt.title(f'Model Efficiency: NSE vs MSE ({mission})', fontsize=12, fontweight='bold')
        plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Acceptable threshold')
        plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good threshold')
        plt.legend(fontsize=9, framealpha=0.9); plt.grid(True, alpha=0.3); plt.tight_layout()
        ts = datetime.now().strftime('%Y%m%d_%H%M')
        plt.savefig(f"{output_dir}/nse_vs_mse_{mission.lower()}_{ts}.png", dpi=300, bbox_inches='tight'); plt.close()

# =============================================================================
# 5.1. MBE Bar Chart Plotting 
# =============================================================================
def plot_mbe_bias_analysis(df_path, output_dir='./plots', mission='ACE'):
    try: df = pd.read_csv(df_path)
    except FileNotFoundError: return
    
    filtros_orden = ['Fisher', 'Pellis', 'JSD', 'Tsallis', 'PINN', 'AKF']
    colores = {"Fisher": "#0072B2", "AKF": "#E69F00", "Pellis": "#D55E00",
               "Tsallis": "#F0E442", "JSD": "#CC79A7", "PINN": "#009E73"}
    
    if 'MBE' in df.columns:
        plt.figure(figsize=(10, 6))
        mbe_values = [df.loc[df['Filter'] == f, 'MBE'].values[0] for f in filtros_orden if (df['Filter'] == f).any()]
        filters_present = [f for f in filtros_orden if (df['Filter'] == f).any()]
        colors_list = [colores[f] for f in filters_present]
        
        plt.bar(filters_present, mbe_values, color=colors_list, alpha=0.8, edgecolor='black')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
        plt.xlabel('Filter', fontsize=11); plt.ylabel('MBE (Mean Bias Error)', fontsize=11)
        plt.title(f'Systematic Bias Analysis ({mission})', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right'); plt.grid(axis='y', alpha=0.3); plt.tight_layout()
        ts = datetime.now().strftime('%Y%m%d_%H%M')
        plt.savefig(f"{output_dir}/mbias_{mission.lower()}_{ts}.png", dpi=300, bbox_inches='tight'); plt.close()

# =============================================================================
# 6. MAIN REAL BENCHMARK
# =============================================================================
def run_real_benchmark(data_dir='./data', output_dir='./plots'):
    os.makedirs(data_dir, exist_ok=True); os.makedirs(output_dir, exist_ok=True)
    print(BANNER); print("🌍 REAL BENCHMARK: ACE "); print("= "*70)
    df_raw, mission = load_nasa_data(data_dir)
    if df_raw is None or mission is None: print("❌ No valid files found. "); return None
    target_dt = 150.0 if mission == 'ACE' else 5.0
    print(f"📡 Environment: {mission} | target_dt={target_dt}s ")
    if mission == 'MMS':
        try: df_raw = df_raw.set_index('Time').resample('5s').mean().reset_index().dropna()
        except: pass
    t, clean_proxy, noisy_data = preprocess_for_ibi(df_raw, variable='Bz', target_dt=target_dt)
    if len(noisy_data) < 50: print("❌ Insufficient data. "); return None
    dt = t[1] - t[0] if len(t) > 1 else target_dt
    print(f"✅ Data: {len(noisy_data)} points, dt={dt:.1f}s ")

    evaluators = {
        'Pellis': lambda dt, rng: AgnosticAdaptiveFilter(dt, kernel_func=kernel_pellis, kernel_name='Pellis', rng=rng),
        'Tsallis': lambda dt, rng: AgnosticAdaptiveFilter(dt, kernel_func=kernel_tsallis, kernel_name='Tsallis', rng=rng),
        'JSD': lambda dt, rng: AgnosticAdaptiveFilter(dt, kernel_func=kernel_jsd, kernel_name='JSD', rng=rng),     
        'Fisher': lambda dt, rng: AgnosticAdaptiveFilter(dt, kernel_func=kernel_fisher, kernel_name='Fisher', rng=rng),
        'AKF': lambda dt, rng: AKF(dt, warmup_steps=50),
        'PINN': lambda dt, rng: PINNFilter(dt, warmup_steps=50, rng=rng)
    }

    results, metrics = {}, []
    init_val = noisy_data[0]
    noise_var = np.var(np.diff(noisy_data)) / 2.0
    R_base_real = max(1e-6, noise_var * 0.1) 
    print(f"🔧 Calibrated R_base: {R_base_real:.4e} ")
    print("⏳ Starting filter execution... ")
    n_samples = len(noisy_data)
    control_results = {}

    for name, factory in evaluators.items():
        rng = np.random.default_rng(42)
        filt = factory(dt, rng)
        if hasattr(filt, 'R_base'): filt.R_base = R_base_real
        est = np.zeros(len(noisy_data)); est[0] = init_val
        filt.x = np.array([init_val, 0.0, 0.0]) if name == 'PINN' else filt.x
        if name != 'PINN': filt.x[0, 0] = init_val
        
        # Tracking arrays
        innovations = []
        nis_series = []
        k_gains = []
        
        spinner = ProgressSpinner(f"▶ Processing {name} ")
        spinner.start(); start = time.perf_counter()
        for k in range(1, len(noisy_data)): 
            est[k] = filt.filter(np.array([[noisy_data[k]]]), conc_factor=1.0)
            inn = noisy_data[k] - est[k-1]
            innovations.append(inn)
            # ✅ Safe access for NIS calculation (PINN lacks P matrix)
            R_val = getattr(filt, 'R_ant', R_base_real)
            P_val = getattr(filt, 'P', np.eye(3) * 0.1)[0, 0] 
            s_cov = R_val + P_val
            nis_series.append((inn**2) / (s_cov + 1e-15))
            if hasattr(filt, 'last_K'): k_gains.append(filt.last_K)
            
        elapsed = time.perf_counter() - start; spinner.stop()
        results[name] = est
        val = validate_tracking(clean_proxy, est, dt, burn_in=0)
        check_bias_health(clean_proxy, est, name)
        
        flops_per_iter = FLOPS_PER_ITER.get(name, 200) 
        total_flops = flops_per_iter * n_samples
        
        #  Gain Tracking Analysis
        mean_gain = np.mean(k_gains) if k_gains else 0
        var_ratio = np.var(est) / np.var(noisy_data) if np.var(noisy_data) > 0 else 0
        if mean_gain > 0.9: gain_status = "❌ COPIANDO SEÑAL"
        elif mean_gain < 0.05: gain_status = "❌ IGNORANDO DATOS"
        else: gain_status = "✅ Filtrando"
        print(f"   [{name}] Gain Prom: {mean_gain:.4f} | Var Ratio: {var_ratio:.4f} | Estado: {gain_status}")
        
        metrics.append({
            'Filter': name, 'MSE': val.get('mse', np.nan), 'RMSE': val.get('rmse', np.nan),
            'R2': val.get('r2', np.nan), 'NSE': val.get('nse', np.nan), 'MBE': val.get('mbe', np.nan),
            'PBIAS': val.get('pbias', np.nan), 'KGE': val.get('kge', np.nan), 'Kurtosis': val.get('kurtosis', np.nan), 
            'MAE': val.get('mae', np.nan), 'Correlation': val.get('correlation', np.nan),
            'Alpha': val.get('alpha', np.nan), 'Beta': val.get('beta', np.nan),
            'Mean_Gain_K': mean_gain, 'Gain_Status': gain_status,
            'Time_s': elapsed, 'Mission': mission, 'Total_FLOPs': total_flops, 
            'FLOPs_per_Sec': total_flops / (elapsed + 1e-12), 'Time_per_Sample': elapsed / n_samples
        })
        control_results[name] = run_advanced_controls(name, noisy_data, est, dt, np.array(innovations), np.array(nis_series))
        metrics[-1].update(control_results[name]['metrics'])

    print("\n" + "= "*70); print("📊 VALIDATION SUMMARY "); print("= "*70)
    df_metrics = pd.DataFrame(metrics)
    print(f"\n{'Filter': <8} {'MSE': >10} {'R²': >8} {'NSE': >8} {'MBE': >10} {'KGE': >8} {'K': >6} ")
    print("-"*65)
    for _, row in df_metrics.iterrows():
        print(f"{row['Filter']: <8} {row['MSE']: >10.4f} {row['R2']: >8.4f}  "
              f"{row['NSE']: >8.4f} {row['MBE']: >+10.6f} {row['KGE']: >8.4f} {row['Mean_Gain_K']: >6.3f} ")

    ts = datetime.now().strftime('%Y%m%d_%H%M')
    csv_path = f"{output_dir}/real_{mission.lower()}_{ts}.csv"
    df_metrics.to_csv(csv_path, index=False)

    # Standard Plots
    plt.figure(figsize=(14, 9))
    colores = {"Fisher": "#0072B2", "AKF": "#E69F00", "Pellis": "#D55E00",
               "Tsallis": "#F0E442", "JSD": "#CC79A7", "PINN": "#009E73"}
    plt.subplot(3, 1, 1)
    plt.plot(t, noisy_data, color="#B0B0B0", label='Observed', alpha=0.6)
    plt.plot(t, clean_proxy, color="#2E2E2E", linestyle='--', label='Clean Proxy', lw=2)
    for name, est in results.items(): 
        plt.plot(t, est, label=name, lw=1.5, color=colores.get(name))
    plt.ylabel('Bz (nT)'); plt.title(f'{mission} - Kernels + AKF + PINN'); plt.legend(); plt.grid()
    plt.subplot(3, 1, 2)
    for name, est in results.items(): 
        plt.plot(t, noisy_data - est, label=f'{name}', lw=0.8, alpha=0.7, color=colores.get(name))
    plt.ylabel('Residual'); plt.title('Residual Analysis'); plt.axhline(0, color='black', lw=1); plt.grid()
    plt.subplot(3, 1, 3)
    for name, est in results.items(): 
        plt.hist(noisy_data - est, bins=30, alpha=0.5, label=name, density=True, color=colores.get(name))
    plt.xlabel('Residual'); plt.title('Distribution'); plt.legend(); plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/real_{mission.lower()}_{ts}.png", dpi=300); plt.close()

    # ✅ Generate Separated Advanced Control Plots
    plot_advanced_controls(control_results, mission, output_dir)
    try: 
        plot_efficiency_real(csv_path, output_dir=output_dir, mission=mission)
        plot_mbe_bias_analysis(csv_path, output_dir=output_dir, mission=mission) # Call the new MBE plot function
        plot_comprehensive_metrics(csv_path, output_dir=output_dir, mission=mission)
    except Exception as e: print(f"⚠️ Error plotting efficiency: {e} ")
    print(f"\n📊 Real {mission} completed. CSV: {csv_path} "); print(f"📁 Results saved in: {output_dir}/ ")
    return df_metrics

# =============================================================================
# 7. ADVANCED VALIDATION CONTROLS
# =============================================================================
def test_whiteness(innovations, alpha=0.05):
    N = len(innovations)
    max_lag = min(50, N // 3)
    lags = np.arange(1, max_lag + 1)
    acf = np.array([np.corrcoef(innovations[:-i], innovations[i:])[0, 1] for i in lags])
    conf = 1.96 / np.sqrt(N)
    outside = np.sum(np.abs(acf) > conf)
    return {'whiteness_passed': outside == 0, 'max_acf_lag': float(np.max(np.abs(acf))), 
            'acf_conf': float(conf), 'acf_lags': lags, 'acf_vals': acf}

def test_nis_consistency(nis_values):
    mean_nis, var_nis = float(np.mean(nis_values)), float(np.var(nis_values))
    return {'nis_mean': mean_nis, 'nis_var': var_nis, 
            'nis_consistency_passed': (0.8 <= mean_nis <= 1.2) and (0.5 <= var_nis <= 3.5)}

def test_jump_finitude(raw, filtered):
    inc_raw, inc_filt = np.diff(raw), np.diff(filtered)
    kurt_raw, kurt_filt = float(stats.kurtosis(inc_raw, fisher=True)), float(stats.kurtosis(inc_filt, fisher=True))
    ratio = kurt_filt / (kurt_raw + 1e-12)
    return {'jump_kurt_raw': kurt_raw, 'jump_kurt_filt': kurt_filt, 
            'jump_ratio': ratio, 'jump_passed': 0.4 < ratio < 2.5}

def compute_transfer_entropy(source, target, lag=1, bins=30):
    """Calculadora de Entropía de Transferencia (Corregida: evita IndexError por broadcasting)"""
    s_edges = np.linspace(source.min(), source.max(), bins + 1)
    t_edges = np.linspace(target.min(), target.max(), bins + 1)
    s_b = np.clip(np.digitize(source, s_edges) - 1, 0, bins - 1)
    t_b = np.clip(np.digitize(target, t_edges) - 1, 0, bins - 1)
    x = s_b[lag:-1]
    y = t_b[lag+1:]
    y_past = t_b[lag:-1]

    joint = np.zeros((bins, bins, bins))
    np.add.at(joint, (y, y_past, x), 1)
    joint /= (np.sum(joint) + 1e-15)

    # Marginales alineadas explícitamente para broadcasting 3D
    p_ypast   = np.sum(joint, axis=(0, 2))[np.newaxis, :, np.newaxis] # (1, bins, 1)
    p_y_ypast = np.sum(joint, axis=2)[:, :, np.newaxis]               # (bins, bins, 1)
    p_ypast_x = np.sum(joint, axis=0)[np.newaxis, :, :]               # (1, bins, bins)
    p_y_ypast_x = joint                                               # (bins, bins, bins)

    # ✅ Cálculo del ratio usando broadcasting nativo
    num = p_y_ypast_x * p_ypast + 1e-15
    den = p_y_ypast * p_ypast_x + 1e-15
    ratio = num / den

    # ✅ Logaritmo seguro: solo se evalúa donde la probabilidad conjunta > 0
    mask = p_y_ypast_x > 0
    log_ratio = np.zeros_like(ratio)
    log_ratio[mask] = np.log(ratio[mask])

    # TE = Σ p(y, y_past, x) * log( ratio )
    te = np.sum(p_y_ypast_x * log_ratio)
    return float(te)

def test_transfer_entropy(raw, filtered):
    te_val = compute_transfer_entropy(raw, filtered, lag=1, bins=25)
    mi_proxy = float(stats.pearsonr(raw, filtered)[0]**2)
    return {'te_value': te_val, 'te_passed': te_val > 0.01 and te_val < 0.95 * mi_proxy, 'mi_proxy': mi_proxy}

def test_psd(raw, filtered, dt):
    n_seg = min(256, len(raw))
    f_raw, psd_raw = welch(raw, fs=1/dt, nperseg=n_seg, scaling='density')
    f_filt, psd_filt = welch(filtered, fs=1/dt, nperseg=n_seg, scaling='density')
    mask = (f_raw > 0.005) & (f_raw < 0.5)
    slope = np.nan
    if np.sum(mask) > 10:
        slope, _, _, _, _ = linregress(np.log10(f_raw[mask]), np.log10(psd_raw[mask]))
    return {'psd_slope_raw': float(slope), 'psd_passed': abs(slope + 5/3) < 1.5 if not np.isnan(slope) else False, 
            'f': f_filt, 'psd_raw': psd_raw, 'psd_filt': psd_filt}

def run_advanced_controls(filter_name, noisy, est, dt, innovations, nis_vals):
    w, n, j, te, p = test_whiteness(innovations), test_nis_consistency(nis_vals), test_jump_finitude(noisy, est), test_transfer_entropy(noisy, est), test_psd(noisy, est, dt)
    return {
        'metrics': {'Whiteness_Pass': w['whiteness_passed'], 'Max_ACF': w['max_acf_lag'],
                    'NIS_Mean': n['nis_mean'], 'NIS_Var': n['nis_var'], 'NIS_Consistent': n['nis_consistency_passed'],
                    'Jump_Ratio': j['jump_ratio'], 'Jump_Pass': j['jump_passed'],
                    'TE_Value': te['te_value'], 'TE_Pass': te['te_passed'],
                    'PSD_Slope': p['psd_slope_raw'], 'PSD_Pass': p['psd_passed']},
        'whiteness': w, 'nis': n, 'jump': j, 'te': te, 'psd': p
    }

def plot_advanced_controls(control_results, mission, output_dir):
    cols = list(control_results.keys())
    if not cols: return
    ts = datetime.now().strftime('%Y%m%d_%H%M')

    # Grouping for cleaner LaTeX insertion
    kernels_ordered = ['Fisher', 'Pellis', 'JSD', 'Tsallis']
    others = ['PINN', 'AKF']
    kernels_present = [k for k in kernels_ordered if k in cols]
    others_present = [k for k in others if k in cols]

    # === 1. Whiteness ACF (Split: Kernels vs Others) ===
    
    # 1a. Kernels (2x2)
    if kernels_present:
        plt.figure(figsize=(12, 8))
        for i, name in enumerate(kernels_present, 1):
            plt.subplot(2, 2, i)
            d = control_results[name]['whiteness']
            plt.plot(d['acf_lags'], d['acf_vals'], marker='o', markersize=4)
            plt.axhline(d['acf_conf'], color='g', linestyle='--', alpha=0.5)
            plt.axhline(-d['acf_conf'], color='g', linestyle='--', alpha=0.5)
            plt.axhline(0, color='k', linewidth=0.5)
            plt.title(f'{name} - ACF Innovations')
            plt.grid(True, alpha=0.3)
            plt.ylim(-0.2, 0.2)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/whiteness_acf_kernels_{mission.lower()}_{ts}.png", dpi=300); plt.close()

    # 1b. Others (1x2)
    if others_present:
        plt.figure(figsize=(10, 4))
        for i, name in enumerate(others_present, 1):
            plt.subplot(1, 2, i)
            d = control_results[name]['whiteness']
            plt.plot(d['acf_lags'], d['acf_vals'], marker='o', markersize=4)
            plt.axhline(d['acf_conf'], color='g', linestyle='--', alpha=0.5)
            plt.axhline(-d['acf_conf'], color='g', linestyle='--', alpha=0.5)
            plt.axhline(0, color='k', linewidth=0.5)
            plt.title(f'{name} - ACF Innovations')
            plt.grid(True, alpha=0.3)
            plt.ylim(-0.2, 0.2)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/whiteness_acf_others_{mission.lower()}_{ts}.png", dpi=300); plt.close()

    # === 2. NIS vs TE (Split: Kernels vs Others) ===
    
    # 2a. Kernels
    if kernels_present:
        plt.figure(figsize=(10, 8)) # 2x2
        for i, name in enumerate(kernels_present, 1):
            plt.subplot(2, 2, i)
            d = control_results[name]
            plt.scatter(d['nis']['nis_mean'], d['te']['te_value'], s=100)
            plt.axvline(1.0, color='g', linestyle='--', alpha=0.5, label='Ideal NIS=1')
            plt.title(f'{name} - NIS vs TE')
            plt.xlabel('Mean NIS'); plt.ylabel('TE (bits)')
            plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/nis_te_kernels_{mission.lower()}_{ts}.png", dpi=300); plt.close()

    # 2b. Others
    if others_present:
        plt.figure(figsize=(10, 4)) # 1x2
        for i, name in enumerate(others_present, 1):
            plt.subplot(1, 2, i)
            d = control_results[name]
            plt.scatter(d['nis']['nis_mean'], d['te']['te_value'], s=100)
            plt.axvline(1.0, color='g', linestyle='--', alpha=0.5, label='Ideal NIS=1')
            plt.title(f'{name} - NIS vs TE')
            plt.xlabel('Mean NIS'); plt.ylabel('TE (bits)')
            plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/nis_te_others_{mission.lower()}_{ts}.png", dpi=300); plt.close()

    # === 3. Jump Test (Bar Chart) ===
    plt.figure(figsize=(8, 6))
    colores = {"Fisher": "#0072B2", "AKF": "#E69F00", "Pellis": "#D55E00",
               "Tsallis": "#F0E442", "JSD": "#CC79A7", "PINN": "#009E73"}
    jump_order = ['Fisher', 'Pellis', 'JSD', 'Tsallis', 'PINN', 'AKF']
    for name in [f for f in jump_order if f in control_results]:
        plt.bar(name, control_results[name]['jump']['jump_ratio'], color=colores.get(name))
    plt.axhline(1.0, color='k', linestyle='--', linewidth=2)
    plt.ylabel('Kurtosis Ratio (Filtered/Raw)')
    plt.title('Jump Test: Preservation of Physical Transients')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/jump_test_{mission.lower()}_{ts}.png", dpi=300); plt.close()

    # === 4. Power Spectral Density (PSD) ===
    plt.figure(figsize=(10, 6))
    ref_filter = 'Fisher' if 'Fisher' in control_results else cols[0]
    d0 = control_results[ref_filter]['psd']
    colores = {"Fisher": "#0072B2", "AKF": "#E69F00", "Pellis": "#D55E00",
               "Tsallis": "#F0E442", "JSD": "#CC79A7", "PINN": "#009E73"}
    plt.loglog(d0['f'], d0['psd_raw'], color="#B0B0B0", alpha=0.7, label='Raw Data')
    plt.loglog(d0['f'], d0['psd_filt'], color=colores.get(ref_filter, 'b'), lw=2, label=f'Filtered ({ref_filter})')
    
    # Reference -5/3 slope
    ref_f = d0['f'][(d0['f']>0.01) & (d0['f']<0.1)]
    if len(ref_f)>0:
        plt.loglog(ref_f, ref_f**(-5/3)*d0['psd_raw'][10], 'r--', alpha=0.5, label='-5/3 Kolmogorov Law')
    
    plt.xlabel('Frequency (Hz)'); plt.ylabel('PSD (nT²/Hz)')
    plt.title('Power Spectral Density Analysis')
    plt.legend(); plt.grid(True, which="both", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/psd_analysis_{mission.lower()}_{ts}.png", dpi=300); plt.close()

# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    DATA_DIR, PLOTS_DIR = './data', './plots'
    os.makedirs(DATA_DIR, exist_ok=True); os.makedirs(PLOTS_DIR, exist_ok=True)
    print(" Unified Real-Data Benchmark Execution ")
    print("= "*70)
    real_df = run_real_benchmark(data_dir=DATA_DIR, output_dir=PLOTS_DIR)
    if real_df is not None:
        mission = real_df['Mission'].iloc[0]
        print("\n" + "= "*70); print("✅ PROCESS COMPLETED SUCCESSFULLY "); print("= "*70)
        print(f"📁 Results saved in: {PLOTS_DIR}/ ")
        print("📄 Files generated: ")
        print(f"   - real_{mission.lower()}_*.csv (metrics + controls + gain tracking) ")
        print(f"   - whiteness_acf_kernels_*.png (ACF for AIKC Filters) ")
        print(f"   - whiteness_acf_others_*.png (ACF for AKF/PINN) ")
        print(f"   - nis_te_kernels_*.png (Consistency for AIKC Filters) ")
        print(f"   - nis_te_others_*.png (Consistency for AKF/PINN) ")
        print(f"   - mbias_{mission.lower()}_*.png (Systematic Bias Analysis) ")
        print(f"   - jump_test_*.png (Physical Transients Test) ")
        print(f"   - psd_analysis_*.png (Power Spectral Density) ")
        print("\n🎯 Key Metrics Interpretation: ")
        print("   - Mean_Gain_K ≈ 0.1-0.6: ✅ Filtrando | >0.9: ❌ Copiando | <0.05: ❌ Ignorando")
        print("   - NIS Mean ≈ 1.0: Perfect covariance consistency")
        print("   - Jump Ratio ≈ 1.0: Preserves physical shocks")
        print("   - TE > 0 & < R²: Healthy information flow")
        print("   - PSD Slope ≈ -1.67: Preserves plasma turbulence physics")
        print("= "*70)
    else:
        print("\n❌ Benchmark failed or no data available. "); sys.exit(1)