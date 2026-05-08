# 🛰️ AIKC-IBI-ACE-Validation

**From Theory to Deep Space: Real-Data Validation of the AIKC Extension for IBI via NASA ACE Mission**

## 📁 Repository Structure
```text
ACE/
├── docs/
│   └── ACE10.pdf                          # Peer-reviewed paper (theory & results)
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
