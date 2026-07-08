import uproot
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter

files = {
    'pos0': '../../01_data/processed/2026_tau_cafe/HistsOutDataCafePos0.root',
    'pos1': '../../01_data/processed/2026_tau_cafe/HistsOutDataCafePos1.root', 
    'sky': '../../01_data/processed/2026_tau_cafe/HistsOutSkyRoofRuns37-77.root'
}

def analyze_and_plot(hist_name, out_prefix, filter_sigma=0):
    f_pos0 = uproot.open(files['pos0'])
    f_pos1 = uproot.open(files['pos1'])
    f_sky = uproot.open(files['sky'])

    h_pos0 = f_pos0[hist_name].values()
    h_pos1 = f_pos1[hist_name].values()
    h_sky = f_sky[hist_name].values()

    # Get axes limits
    ax0_edges = f_sky[hist_name].axes[0].edges()
    ax1_edges = f_sky[hist_name].axes[1].edges()
    extent = [ax0_edges[0], ax0_edges[-1], ax1_edges[0], ax1_edges[-1]]

    # Normalize by total hits
    norm_pos0 = h_pos0 / np.sum(h_pos0)
    norm_pos1 = h_pos1 / np.sum(h_pos1)
    norm_sky = h_sky / np.sum(h_sky)

    # Avoid division by zero
    mask = norm_sky > (np.max(norm_sky) * 1e-4) # exclude areas with very few sky muons
    
    ratio_pos0 = np.zeros_like(norm_pos0)
    ratio_pos1 = np.zeros_like(norm_pos1)

    ratio_pos0[mask] = norm_pos0[mask] / norm_sky[mask]
    ratio_pos1[mask] = norm_pos1[mask] / norm_sky[mask]

    # Optional filtering
    if filter_sigma > 0:
        ratio_pos0 = gaussian_filter(ratio_pos0, sigma=filter_sigma)
        ratio_pos1 = gaussian_filter(ratio_pos1, sigma=filter_sigma)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = ax1.imshow(ratio_pos0.T, origin='lower', extent=extent, cmap='viridis', vmin=0.5, vmax=1.5)
    ax1.set_title(f'Pos 0 Transmission (Ratio)')
    if hist_name.startswith('XY'):
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
    else:
        ax1.set_xlabel('Tx')
        ax1.set_ylabel('Ty')
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(ratio_pos1.T, origin='lower', extent=extent, cmap='viridis', vmin=0.5, vmax=1.5)
    ax2.set_title(f'Pos 1 Transmission (Ratio)')
    if hist_name.startswith('XY'):
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
    else:
        ax2.set_xlabel('Tx')
        ax2.set_ylabel('Ty')
    fig.colorbar(im2, ax=ax2)

    plt.suptitle(f'Muon Transmission - {hist_name} (Filter sigma={filter_sigma})')
    plt.tight_layout()
    plt.savefig(f'{out_prefix}.png', dpi=150)
    plt.close()
    print(f'Saved {out_prefix}.png')

# Experiment with different histograms and smoothing
# analyze_and_plot('txtyN;1', 'ratio_txtyN_raw', filter_sigma=0)
# analyze_and_plot('txtyN;1', 'ratio_txtyN_smooth', filter_sigma=1.0)
# analyze_and_plot('txty;1', 'ratio_txty_raw', filter_sigma=0)
# analyze_and_plot('txty;1', 'ratio_txty_smooth2', filter_sigma=2.0)
# analyze_and_plot('txty;1', 'ratio_txty_smooth5', filter_sigma=5.0)

analyze_and_plot('XY07m;1', 'ratio_XY07m_raw', filter_sigma=0)
analyze_and_plot('XY07m;1', 'ratio_XY07m_smooth2', filter_sigma=2.0)
analyze_and_plot('XY07m;1', 'ratio_XY07m_smooth5', filter_sigma=5.0)
analyze_and_plot('XY07m;1', 'ratio_XY07m_smooth10', filter_sigma=10.0)


