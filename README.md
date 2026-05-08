# 🛰️ AIKC-IBI-ACE-Validation

**From Theory to Deep Space: Real-Data Validation of the AIKC Extension for IBI via NASA ACE Mission**

## 📁 Repository Structure
```text
ACE/
├── docs/
│   └── ACE10.pdf                          # Paper (theory & results)
├── python/
│   ├── aikc_ibi_nasa_ace_validation.py    # Core benchmarking script
│   ├── data/                              # ✅ Raw NASA ACE .cdf files (pre-included)
│   │   ├── ac_h2_mfi_20230101_v01.cdf
│   │   ├── ac_h2_mfi_20230102_v01.cdf
│   │   └── ... (all 2023 MFI Level-2 files)
│   └── plots/                             # Auto-generated outputs
│       ├── real_ace_*.csv                 # Comprehensive metrics report
│       ├── real_ace_*.png                 # Temporal traces & residuals
│       ├── eficiencia_real_*.png          # MSE vs FLOPs efficiency plot
│       ├── mbias_*.png                    # Systematic bias (MBE) bar chart
│       ├── nse_vs_mse_*.png               # Model efficiency scatter
│       ├── whiteness_acf_kernels_*.png    # ACF for AIKC kernels
│       ├── whiteness_acf_others_*.png     # ACF for AKF/PINN
│       ├── nis_te_kernels_*.png           # NIS vs Transfer Entropy (AIKC)
│       ├── nis_te_others_*.png            # NIS vs Transfer Entropy (AKF/PINN)
│       ├── jump_test_*.png                # Physical transients preservation
│       └── psd_analysis_*.png             # Power Spectral Density analysis
├── README.md                              # ← This file
├── LICENSE                                # GNU GPL v3.0
└── requirements.txt                       # Python dependencies

▶️ What Happens When You Run the Script?
    Auto-discovers all .cdf files in python/data/
    Parses epochs with automatic unit detection (ms, s, or relative)
    Filters & resamples data to uniform 150 s cadence
    Executes AIKC (Fisher, Pellis, JSD, Tsallis), AKF, and PINN filters
    Computes 15+ validation metrics: MSE, NSE, MBE, KGE, Whiteness, NIS, Jump Test, Transfer Entropy, PSD, Filter Gain, FLOPs
    Exports publication-ready figures and a comprehensive .csv report to python/plots/

📡 Data & Reproducibility
✅ Pre-included Dataset (Ready-to-Run)
All raw NASA ACE Magnetic Field Instrument (MFI) Level-2 .cdf files for the year 2023 are pre-included in the python/data/ directory. No external downloads or manual data preparation are required for baseline validation.

    Mission: NASA Advanced Composition Explorer (ACE)
    Instrument: Magnetic Field Instrument (MFI)
    Product: Level-2 (ac_h2_mfi_*.cdf)
    Variable: Bz (Z-component of interplanetary magnetic field in GSM coordinates)
    Period: January 1, 2023 – December 31, 2023
    Samples: 210,216 high-resolution measurements
    Target Cadence: 150 seconds

➕ Adding More Data or Testing Different Date Ranges
To extend the analysis or validate the framework with different time periods:

    Download additional .cdf files from the official NASA SPDF CDAWeb repository:
https://spdf.gsfc.nasa.gov/pub/data/ace/mag/level_2_cdaweb/mfi_h2/

        Navigate to the desired year folder (e.g., 2022/, 2024/)
        Download any ac_h2_mfi_*.cdf files corresponding to your period of interest
    Place the files directly into the python/data/ directory:

ACE/
└── python/
    └── data/
        ├── ac_h2_mfi_20230101_v01.cdf  # (pre-included)
        ├── ac_h2_mfi_20230102_v01.cdf  # (pre-included)
        ├── ac_h2_mfi_20240101_v01.cdf  # (your added file)
        └── ...
    Run the script as usual:
    bash
cd python
python aikc_ibi_nasa_ace_validation.py

🔹 The script automatically:
    Detects and loads all .cdf files present in python/data/
    Parses epochs with automatic unit detection (ms, s, or relative)
    Filters data within the valid range (2010–2026)
    Resamples to the target cadence (150 s for ACE)
    Processes the combined dataset without requiring code modifications
 
 Primary Source: NASA ACE Mission MFI Level-2 Product  
    Official Repository: SPDF CDAWeb - ACE/MFI    
    Variable Used: Bz (Z-component of interplanetary magnetic field in GSM coordinates)  
    Citation: Smith CW, ACE MFI Team. ACE Magnetic Field Instrument (MFI) Level 2 Data. NASA Space Physics Data Facility (SPDF); 2023.

Theoretical Framework
Effective IBI Theory (Baseline)
Information-Based Instrumentation (IBI) rests on four invariant mathematical pillars:
1.	Interaction Hamiltonian: Ion-dipolar coupling of sensor with medium
2.	Mori–Zwanzig Projection: Temporal memory kernel separating relevant/irrelevant variables
3.	Fisher Geometry: Local distinguishability and Cramér–Rao bound
4.	3-State Kalman Filter: Augmented state space encoding concentration, trend, and memory
AIKC Extension (Modular Adaptive Layer)
AIKC introduces a contractive adaptive control layer that dynamically scales the measurement covariance:
R_eff,k = clip( R₀ [1 + β σ( (K[I_k] - K₀) / δ )], R_min, R_max )
where:
•	I_k = innovation buffer (sliding window, B=64)
•	K[·] = information functional evaluated on normalized histogram (15 bins)
•	σ(·) = logistic function guaranteeing contractive mapping
•	β, δ, K₀ = empirically calibrated parameters (warmup phase)
Implemented Information Kernels
Kernel	Computational Form	Physical Domain
Fisher (Proxy)	Σ(Δpᵢ)² / Σpᵢ	Local curvature, loss of distinguishability
Pellis (Fractal)	-Σ pᵢ log_φ(pᵢ/πᵢ)	Self-similarity, scale correlations
Jensen–Shannon	½ D_KL(p‖m) + ½ D_KL(π‖m)	Symmetric divergence, multimodality
Tsallis-q (q=1.5)	(1 - Σ pᵢ^q) / (q-1)	Heavy tails, kappa distributions
All kernels operate on normalized probabilities pᵢ > 10⁻¹² to avoid numerical singularities.
📊 Validation Metrics Explained
Metric	Description	Ideal Value
MSE	Mean Squared Error	→ 0
NSE	Nash–Sutcliffe Efficiency	→ 1.0
MBE	Mean Bias Error (systematic bias)	→ 0
KGE	Kling–Gupta Efficiency	→ 1.0
Kurtosis	Tail heaviness of residuals	Matches physical process
Jump Ratio	Kurtosis(filtered)/Kurtosis(raw) increments	≈ 1.0 (preserves transients)
Whiteness	Max autocorrelation of innovations	< confidence bound
NIS Mean	Normalized Innovation Square consistency	≈ 1.0
Transfer Entropy	Information flow from observations to estimates	> 0, < R²
PSD Slope	Power Spectral Density scaling	≈ -5/3 (Kolmogorov)
Mean Gain K	Average Kalman gain magnitude	0.1–0.6 (active filtering)
FLOPs	Floating-point operations (computational cost)	Lower = more edge-viable

🏢 Commercial & Private Licensing
This project is distributed under the GNU General Public License v3.0 for open-source, academic, and non-commercial use.

⚠️ For Companies & Proprietary Use:
Organizations that do not wish to comply with the copyleft obligations of the GNU GPL (e.g., closed-source distribution, commercial SaaS integration, proprietary aerospace deployment, or integration into commercial satellite systems) must contact the author directly to acquire a private/commercial license. Dual-licensing options are available for enterprise, defense, and commercial space applications.

📧 Contact for Commercial Licensing:
dvivas1@uc.edu.ve | vivaszdaniel@gmail.com
