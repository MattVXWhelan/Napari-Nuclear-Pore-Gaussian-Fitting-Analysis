"""
Napari Nuclear Pore Gaussian Fitting Analysis
==============================================

This script detects nuclear pores and fits 2D Gaussians to each pore
to calculate Full Width Half Maximum (FWHM) measurements.

Required Packages:
------------------
pip install napari[all] pyqt5 scikit-image scipy numpy pandas matplotlib seaborn

Features:
---------
1. Background subtraction (tophat, rolling ball, or none)
2. Multiple thresholding methods for pore detection
3. 2D Gaussian fitting to each detected pore
4. FWHM calculation in both X and Y directions
5. Comprehensive visualization of results
6. Detailed CSV output with fit parameters

Usage:
------
1. Run this script: python Nuclear_Pore_Gaussian_Analysis.py
2. Adjust background subtraction and threshold parameters
3. Click "Run Analysis"
4. Results include:
   - Per-pore measurements with Gaussian fit parameters
   - FWHM distributions
   - Visualization of fitted Gaussians
"""

import napari
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import curve_fit
from skimage import filters, measure, morphology, restoration
from skimage.feature import blob_log, blob_dog, blob_doh
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from magicgui import magicgui
import warnings
warnings.filterwarnings('ignore')


def generate_synthetic_pore_data(size=(1024, 1024), n_pores=200, seed=42):
    """Generate realistic synthetic nuclear pore channel data"""
    np.random.seed(seed)
    
    pores = np.zeros(size, dtype=float)
    
    for i in range(n_pores):
        y = np.random.randint(20, size[0] - 20)
        x = np.random.randint(20, size[1] - 20)
        
        # Variable pore size and intensity
        sigma_x = np.random.uniform(1.5, 3.5)
        sigma_y = np.random.uniform(1.5, 3.5)
        amplitude = np.random.uniform(150, 250)
        
        # Create 2D Gaussian
        yy, xx = np.ogrid[:size[0], :size[1]]
        gaussian = amplitude * np.exp(
            -((xx - x)**2 / (2 * sigma_x**2) + 
              (yy - y)**2 / (2 * sigma_y**2))
        )
        pores += gaussian
    
    # Add noise
    pores += np.random.poisson(8, size)
    pores += np.random.normal(15, 3, size)
    
    # Add uneven background
    yy, xx = np.meshgrid(np.linspace(0, 1, size[0]), np.linspace(0, 1, size[1]), indexing='ij')
    background = 30 * (0.5 + 0.3 * np.sin(2 * np.pi * xx) + 0.2 * np.cos(2 * np.pi * yy))
    pores += background
    
    pores = np.clip(pores, 0, 255)
    
    return pores


def apply_background_subtraction(image, method='none', **params):
    """
    Apply background subtraction to image
    
    Parameters:
    -----------
    image : ndarray
        Input image
    method : str
        Background subtraction method:
        - 'none': No background subtraction
        - 'tophat': Morphological top-hat (white top-hat)
        - 'rolling_ball': Rolling ball algorithm
    **params : dict
        Method-specific parameters:
        - tophat_size: disk size for top-hat (default: 15)
        - rolling_ball_radius: radius for rolling ball (default: 50)
    
    Returns:
    --------
    ndarray : Background-subtracted image
    """
    if method == 'none':
        return image
    
    elif method == 'tophat':
        size = params.get('tophat_size', 15)
        selem = morphology.disk(size)
        return morphology.white_tophat(image, selem)
    
    elif method == 'rolling_ball':
        radius = params.get('rolling_ball_radius', 50)
        background = restoration.rolling_ball(image, radius=radius)
        result = image.astype(np.float32) - background
        result = np.clip(result, 0, None)
        return result.astype(image.dtype)
    
    else:
        print(f"Unknown background subtraction method: {method}")
        return image


def apply_threshold(image, method='otsu', manual_threshold=None):
    """Apply thresholding to image"""
    if method == 'manual':
        if manual_threshold is None:
            raise ValueError("manual_threshold must be provided for manual method")
        return manual_threshold
    
    threshold_funcs = {
        'otsu': filters.threshold_otsu,
        'li': filters.threshold_li,
        'triangle': filters.threshold_triangle,
        'yen': filters.threshold_yen,
        'isodata': filters.threshold_isodata,
        'mean': filters.threshold_mean,
        'minimum': filters.threshold_minimum,
    }
    
    if method not in threshold_funcs:
        print(f"Unknown threshold method: {method}, using otsu")
        method = 'otsu'
    
    try:
        threshold = threshold_funcs[method](image)
        return threshold
    except Exception as e:
        print(f"Error computing {method} threshold: {e}")
        print("Falling back to Otsu's method")
        return filters.threshold_otsu(image)


def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    2D Gaussian function for fitting
    
    Parameters:
    -----------
    xy : tuple of arrays
        x and y coordinates
    amplitude : float
        Peak amplitude
    xo, yo : float
        Center position
    sigma_x, sigma_y : float
        Standard deviations in x and y
    theta : float
        Rotation angle (radians)
    offset : float
        Background offset
    
    Returns:
    --------
    1D array : Flattened Gaussian values
    """
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2) / (2 * sigma_x**2) + (np.sin(theta)**2) / (2 * sigma_y**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta)**2) / (2 * sigma_x**2) + (np.cos(theta)**2) / (2 * sigma_y**2)
    g = offset + amplitude * np.exp(-(a * ((x - xo)**2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo)**2)))
    return g.ravel()


def fit_gaussian_to_pore(image, centroid, window_size=15):
    """
    Fit a 2D Gaussian to a nuclear pore
    
    Parameters:
    -----------
    image : ndarray
        Image data
    centroid : tuple
        (y, x) coordinates of pore center
    window_size : int
        Size of window around centroid for fitting
    
    Returns:
    --------
    dict : Fit parameters and quality metrics
    """
    y_center, x_center = int(centroid[0]), int(centroid[1])
    half_window = window_size // 2
    
    # Extract window around pore
    y_min = max(0, y_center - half_window)
    y_max = min(image.shape[0], y_center + half_window + 1)
    x_min = max(0, x_center - half_window)
    x_max = min(image.shape[1], x_center + half_window + 1)
    
    window = image[y_min:y_max, x_min:x_max]
    
    # Create coordinate arrays
    y = np.arange(y_min, y_max)
    x = np.arange(x_min, x_max)
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Find actual peak location in window (not just the detected centroid)
    peak_idx = np.unravel_index(np.argmax(window), window.shape)
    xo_guess = x_min + peak_idx[1]
    yo_guess = y_min + peak_idx[0]
    
    # Initial guesses
    amplitude_guess = window.max() - window.min()
    sigma_guess = 2.0
    theta_guess = 0.0
    offset_guess = window.min()
    
    initial_guess = (amplitude_guess, xo_guess, yo_guess, 
                     sigma_guess, sigma_guess, theta_guess, offset_guess)
    
    # Bounds for parameters
    bounds_lower = [0, x_min, y_min, 0.5, 0.5, -np.pi, 0]
    bounds_upper = [np.inf, x_max, y_max, 10, 10, np.pi, np.inf]
    
    try:
        # Fit Gaussian
        popt, pcov = curve_fit(
            gaussian_2d, 
            (x_grid, y_grid), 
            window.ravel(),
            p0=initial_guess,
            bounds=(bounds_lower, bounds_upper),
            maxfev=5000
        )
        
        amplitude, xo, yo, sigma_x, sigma_y, theta, offset = popt
        
        # Calculate FWHM
        # FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.355 * sigma
        fwhm_x = 2.355 * sigma_x
        fwhm_y = 2.355 * sigma_y
        fwhm_mean = (fwhm_x + fwhm_y) / 2
        
        # Calculate R-squared (goodness of fit)
        fitted_data = gaussian_2d((x_grid, y_grid), *popt).reshape(window.shape)
        residuals = window - fitted_data
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((window - np.mean(window))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Standard errors from covariance matrix
        perr = np.sqrt(np.diag(pcov))
        
        return {
            'success': True,
            'amplitude': amplitude,
            'x_center': xo,
            'y_center': yo,
            'sigma_x': sigma_x,
            'sigma_y': sigma_y,
            'theta': theta,
            'offset': offset,
            'fwhm_x': fwhm_x,
            'fwhm_y': fwhm_y,
            'fwhm_mean': fwhm_mean,
            'r_squared': r_squared,
            'amplitude_err': perr[0],
            'sigma_x_err': perr[3],
            'sigma_y_err': perr[4],
            'fitted_window': fitted_data,
            'residuals': residuals,
            'window': window,
            'window_coords': (y_min, y_max, x_min, x_max)
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'amplitude': np.nan,
            'x_center': x_center,
            'y_center': y_center,
            'sigma_x': np.nan,
            'sigma_y': np.nan,
            'theta': np.nan,
            'offset': np.nan,
            'fwhm_x': np.nan,
            'fwhm_y': np.nan,
            'fwhm_mean': np.nan,
            'r_squared': np.nan,
            'amplitude_err': np.nan,
            'sigma_x_err': np.nan,
            'sigma_y_err': np.nan,
        }


class NuclearPoreGaussianAnalyzer:
    """Analyzer for nuclear pore detection and Gaussian fitting"""
    
    def __init__(self, viewer):
        self.viewer = viewer
        self.results = []
    
    def detect_pores(self, image, method='log', **params):
        """Detect nuclear pores using various methods"""
        if method in ['log', 'dog', 'doh']:
            # Filter parameters for blob detection methods
            blob_params = {k: v for k, v in params.items() 
                          if k in ['min_sigma', 'max_sigma', 'num_sigma', 'threshold', 'overlap']}
            return self._detect_blobs(image, method, **blob_params)
        elif method == 'threshold':
            # Filter parameters for threshold detection method
            threshold_params = {k: v for k, v in params.items() 
                               if k in ['threshold_method', 'manual_threshold', 'min_size', 'max_size', 'smoothing']}
            return self._detect_threshold(image, **threshold_params)
        else:
            raise ValueError(f"Unknown detection method: {method}")
    
    def _detect_blobs(self, image, method, min_sigma=1, max_sigma=5, 
                     num_sigma=5, threshold=0.1, overlap=0.5):
        """Detect blobs using LoG/DoG/DoH"""
        print(f"   Detecting pores using {method.upper()} method...")
        
        # Normalize image to 0-1 range for blob detection (CRITICAL!)
        image_norm = (image - image.min()) / (image.max() - image.min() + 1e-10)
        
        if method == 'log':
            blobs = blob_log(image_norm, min_sigma=min_sigma, max_sigma=max_sigma,
                           num_sigma=num_sigma, threshold=threshold, overlap=overlap)
        elif method == 'dog':
            blobs = blob_dog(image_norm, min_sigma=min_sigma, max_sigma=max_sigma,
                           threshold=threshold, overlap=overlap)
        elif method == 'doh':
            blobs = blob_doh(image_norm, min_sigma=min_sigma, max_sigma=max_sigma,
                           num_sigma=num_sigma, threshold=threshold)
        
        if len(blobs) == 0:
            return np.array([])
        
        coords = blobs[:, :2]
        print(f"   ✓ Detected {len(coords)} pores")
        return coords
    
    def _detect_threshold(self, image, threshold_method='li', manual_threshold=100,
                         min_size=3, max_size=200, smoothing=0.5):
        """Detect pores using threshold-based segmentation"""
        print(f"   Detecting pores using threshold method ({threshold_method})...")
        
        # Smooth image
        if smoothing > 0:
            smoothed = filters.gaussian(image, sigma=smoothing)
        else:
            smoothed = image
        
        # Apply threshold
        thresh = apply_threshold(smoothed, method=threshold_method, 
                                manual_threshold=manual_threshold)
        binary = smoothed > thresh
        
        # Remove small and large objects
        binary = morphology.remove_small_objects(binary, min_size=min_size)
        labels = measure.label(binary)
        
        # Filter by size and extract centroids
        regions = measure.regionprops(labels)
        coords = []
        
        for region in regions:
            if min_size <= region.area <= max_size:
                coords.append(region.centroid)
        
        if len(coords) == 0:
            return np.array([])
        
        coords = np.array(coords)
        print(f"   ✓ Detected {len(coords)} pores")
        return coords
    
    def analyze_pores(self, pore_img, bg_subtract_method='none', 
                     bg_subtract_params=None, detection_method='log',
                     detection_params=None, fit_window_size=15, pixel_size_nm=65.0,
                     min_r_squared=0.7, min_amplitude=20):
        """
        Complete analysis pipeline: detection → Gaussian fitting
        
        Parameters:
        -----------
        pore_img : ndarray
            Nuclear pore channel image
        bg_subtract_method : str
            Background subtraction method
        bg_subtract_params : dict
            Parameters for background subtraction
        detection_method : str
            Pore detection method
        detection_params : dict
            Parameters for pore detection
        fit_window_size : int
            Window size for Gaussian fitting
        pixel_size_nm : float
            Physical size of one pixel in nanometers
        min_r_squared : float
            Minimum R² for quality filtering
        min_amplitude : int
            Minimum amplitude for quality filtering
        
        Returns:
        --------
        pd.DataFrame : Results with Gaussian fit parameters
        list : Detailed fit results for each pore
        """
        print("\n" + "="*60)
        print("NUCLEAR PORE GAUSSIAN FITTING ANALYSIS")
        print("="*60)
        
        # Apply background subtraction
        if bg_subtract_params is None:
            bg_subtract_params = {}
        
        print(f"\n📍 Applying background subtraction: {bg_subtract_method}")
        pore_corrected = apply_background_subtraction(
            pore_img,
            method=bg_subtract_method,
            **bg_subtract_params
        )
        
        # Add corrected image to viewer
        if bg_subtract_method != 'none':
            try:
                # Remove old layer if it exists
                if 'Pores_BG_Subtracted' in [layer.name for layer in self.viewer.layers]:
                    self.viewer.layers.remove('Pores_BG_Subtracted')
                
                self.viewer.add_image(
                    pore_corrected,
                    name='Pores_BG_Subtracted',
                    colormap='magma',
                    visible=True,
                    opacity=0.7
                )
                print(f"   ✓ Added 'Pores_BG_Subtracted' layer (magma)")
            except Exception as e:
                print(f"   Warning: Could not add BG subtracted layer: {e}")
        
        # Detect pores
        if detection_params is None:
            detection_params = {}
        
        pore_coords = self.detect_pores(
            pore_corrected,
            method=detection_method,
            **detection_params
        )
        
        if len(pore_coords) == 0:
            print("✗ No pores detected!")
            return pd.DataFrame(), []
        
        # Check for mask from shapes layer
        mask_info = None
        shapes_layers = [layer for layer in self.viewer.layers if isinstance(layer, napari.layers.Shapes)]
        if len(shapes_layers) > 0:
            # Use the first shapes layer as mask
            shapes_layer = shapes_layers[0]
            if len(shapes_layer.data) > 0:
                print(f"\n🎯 Found {len(shapes_layer.data)} shape(s) - filtering pores...")
                # Create individual masks for each shape
                masks = shapes_layer.to_masks(mask_shape=pore_img.shape)
                
                # Assign each pore to a mask/cell
                pores_before = len(pore_coords)
                mask_assignments = []
                pores_to_keep = []
                
                for coord in pore_coords:
                    y, x = int(coord[0]), int(coord[1])
                    if 0 <= y < masks.shape[1] and 0 <= x < masks.shape[2]:
                        # Check which mask(s) contain this pore
                        for mask_idx in range(masks.shape[0]):
                            if masks[mask_idx, y, x]:
                                mask_assignments.append(mask_idx + 1)  # 1-indexed for user
                                pores_to_keep.append(True)
                                break
                        else:
                            mask_assignments.append(0)  # Not in any mask
                            pores_to_keep.append(False)
                    else:
                        mask_assignments.append(0)
                        pores_to_keep.append(False)
                
                # Filter to only pores inside masks
                pore_coords = pore_coords[pores_to_keep]
                mask_assignments = [m for m, keep in zip(mask_assignments, pores_to_keep) if keep]
                
                print(f"   ✓ Kept {len(pore_coords)}/{pores_before} pores inside mask(s)")
                
                if len(pore_coords) == 0:
                    print("✗ No pores inside mask!")
                    return pd.DataFrame(), []
                
                # Calculate area of each mask
                mask_areas_pixels = []
                mask_areas_um2 = []
                pixel_area_um2 = (pixel_size_nm / 1000) ** 2  # Convert nm to µm
                
                for mask_idx in range(masks.shape[0]):
                    area_pixels = np.sum(masks[mask_idx])
                    area_um2 = area_pixels * pixel_area_um2
                    mask_areas_pixels.append(area_pixels)
                    mask_areas_um2.append(area_um2)
                
                mask_info = {
                    'assignments': mask_assignments,
                    'n_masks': masks.shape[0],
                    'areas_pixels': mask_areas_pixels,
                    'areas_um2': mask_areas_um2
                }
        
        # Fit Gaussians to each pore
        print(f"\n🔬 Fitting 2D Gaussians to {len(pore_coords)} pores...")
        fit_results = []
        
        for i, coord in enumerate(pore_coords):
            if (i + 1) % 50 == 0:
                print(f"   Progress: {i+1}/{len(pore_coords)} pores fitted")
            
            fit_result = fit_gaussian_to_pore(
                pore_corrected,
                coord,
                window_size=fit_window_size
            )
            
            fit_result['pore_id'] = i + 1
            fit_result['detected_y'] = coord[0]
            fit_result['detected_x'] = coord[1]
            
            # Add mask ID if using ROI masking
            if mask_info is not None:
                fit_result['mask_id'] = mask_info['assignments'][i]
            else:
                fit_result['mask_id'] = 0  # No mask used
            
            fit_results.append(fit_result)
        
        # Create summary dataframe
        successful_fits = [r for r in fit_results if r['success']]
        failed_fits = [r for r in fit_results if not r['success']]
        
        print(f"\n✓ Gaussian fitting complete!")
        print(f"   Successful fits: {len(successful_fits)}/{len(fit_results)}")
        print(f"   Failed fits: {len(failed_fits)}/{len(fit_results)}")
        
        # Create dataframe
        df_data = []
        for result in fit_results:
            fwhm_max_pixels = max(result['fwhm_x'], result['fwhm_y'])
            fwhm_max_nm = fwhm_max_pixels * pixel_size_nm
            
            df_data.append({
                'pore_id': result['pore_id'],
                'mask_id': result['mask_id'],
                'detected_y': result['detected_y'],
                'detected_x': result['detected_x'],
                'fit_success': result['success'],
                'amplitude': result['amplitude'],
                'fitted_y': result['y_center'],
                'fitted_x': result['x_center'],
                'sigma_x_pixels': result['sigma_x'],
                'sigma_y_pixels': result['sigma_y'],
                'sigma_x_nm': result['sigma_x'] * pixel_size_nm,
                'sigma_y_nm': result['sigma_y'] * pixel_size_nm,
                'theta_rad': result['theta'],
                'theta_deg': np.rad2deg(result['theta']) if result['success'] else np.nan,
                'offset': result['offset'],
                'fwhm_x_pixels': result['fwhm_x'],
                'fwhm_y_pixels': result['fwhm_y'],
                'fwhm_mean_pixels': result['fwhm_mean'],
                'fwhm_max_pixels': fwhm_max_pixels,
                'fwhm_x_nm': result['fwhm_x'] * pixel_size_nm,
                'fwhm_y_nm': result['fwhm_y'] * pixel_size_nm,
                'fwhm_mean_nm': result['fwhm_mean'] * pixel_size_nm,
                'fwhm_max_nm': fwhm_max_nm,
                'r_squared': result['r_squared'],
                'amplitude_err': result['amplitude_err'],
                'sigma_x_err_pixels': result['sigma_x_err'],
                'sigma_y_err_pixels': result['sigma_y_err'],
                'sigma_x_err_nm': result['sigma_x_err'] * pixel_size_nm,
                'sigma_y_err_nm': result['sigma_y_err'] * pixel_size_nm,
            })
        
        df = pd.DataFrame(df_data)
        
        # Print summary statistics
        if len(successful_fits) > 0:
            # Filter for high quality fits using provided thresholds
            df_high_quality = df[(df['fit_success']) & 
                                (df['r_squared'] > min_r_squared) & 
                                (df['amplitude'] > min_amplitude)]
            
            print(f"\n📊 FWHM Statistics:")
            print(f"   Pixel size: {pixel_size_nm:.2f} nm/pixel")
            print(f"   Quality filters: R² > {min_r_squared}, amplitude > {min_amplitude}")
            print(f"\n   All successful fits ({len(successful_fits)}):")
            print(f"   Mean FWHM: {df[df['fit_success']]['fwhm_mean_pixels'].mean():.3f} ± {df[df['fit_success']]['fwhm_mean_pixels'].std():.3f} pixels")
            print(f"              {df[df['fit_success']]['fwhm_mean_nm'].mean():.2f} ± {df[df['fit_success']]['fwhm_mean_nm'].std():.2f} nm")
            print(f"   Max FWHM:  {df[df['fit_success']]['fwhm_max_pixels'].mean():.3f} ± {df[df['fit_success']]['fwhm_max_pixels'].std():.3f} pixels")
            print(f"              {df[df['fit_success']]['fwhm_max_nm'].mean():.2f} ± {df[df['fit_success']]['fwhm_max_nm'].std():.2f} nm")
            print(f"   Mean R²: {df[df['fit_success']]['r_squared'].mean():.4f}")
            
            if len(df_high_quality) > 0:
                print(f"\n   High quality fits only (n={len(df_high_quality)}):")
                print(f"   Mean FWHM: {df_high_quality['fwhm_mean_pixels'].mean():.3f} ± {df_high_quality['fwhm_mean_pixels'].std():.3f} pixels")
                print(f"              {df_high_quality['fwhm_mean_nm'].mean():.2f} ± {df_high_quality['fwhm_mean_nm'].std():.2f} nm")
                print(f"   Max FWHM:  {df_high_quality['fwhm_max_pixels'].mean():.3f} ± {df_high_quality['fwhm_max_pixels'].std():.3f} pixels")
                print(f"              {df_high_quality['fwhm_max_nm'].mean():.2f} ± {df_high_quality['fwhm_max_nm'].std():.2f} nm")
                print(f"   Mean R²: {df_high_quality['r_squared'].mean():.4f}")
            
            # Print per-mask statistics if masks were used
            if mask_info is not None and mask_info['n_masks'] >= 1:
                print(f"\n   📍 Per-Mask/Cell Statistics (high quality only):")
                for mask_id in range(1, mask_info['n_masks'] + 1):
                    mask_data = df_high_quality[df_high_quality['mask_id'] == mask_id]
                    if len(mask_data) > 0:
                        median_fwhm = mask_data['fwhm_max_nm'].median()
                        q1 = mask_data['fwhm_max_nm'].quantile(0.25)
                        q3 = mask_data['fwhm_max_nm'].quantile(0.75)
                        iqr = q3 - q1
                        area_um2 = mask_info['areas_um2'][mask_id - 1]
                        pore_density = len(mask_data) / area_um2
                        print(f"   Mask/Cell {mask_id}:")
                        print(f"      Area: {area_um2:.2f} µm²")
                        print(f"      Pores: {len(mask_data)} ({pore_density:.3f} pores/µm²)")
                        print(f"      Max FWHM - Median: {median_fwhm:.2f} nm, IQR: {iqr:.2f} nm (Q1={q1:.2f}, Q3={q3:.2f})")
        
        # Add pore locations to viewer
        try:
            # Remove old layer if it exists
            layer_names = [layer.name for layer in self.viewer.layers]
            if 'Detected Pores' in layer_names:
                self.viewer.layers.remove('Detected Pores')
            
            # Color by fit success (green = success, red = failed)
            face_colors = ['cyan' if r['success'] else 'red' for r in fit_results]
            
            # Add points layer with minimal parameters
            points_layer = self.viewer.add_points(
                pore_coords,
                name='Detected Pores',
                size=5,
                face_color=face_colors
            )
            print(f"\n✓ Added 'Detected Pores' layer to viewer ({len(pore_coords)} pores)")
            print(f"   Cyan = successful fits ({len([c for c in face_colors if c == 'cyan'])})")
            print(f"   Red = failed fits ({len([c for c in face_colors if c == 'red'])})")
        except Exception as e:
            print(f"\n✗ Warning: Could not add points layer to viewer: {e}")
            import traceback
            traceback.print_exc()
        
        return df, fit_results, mask_info
    
    def plot_results(self, df, fit_results, pore_img, mask_info=None, min_r_squared=0.7, min_amplitude=20):
        """Create comprehensive visualization of results"""
        # Filter for successful fits with quality thresholds
        df_success = df[(df['fit_success']) & 
                       (df['r_squared'] > min_r_squared) & 
                       (df['amplitude'] > min_amplitude)].copy()
        
        if len(df_success) == 0:
            print(f"No successful fits with R² > {min_r_squared} and amplitude > {min_amplitude} to plot!")
            return None
        
        excluded_r2 = len(df[(df['fit_success']) & (df['r_squared'] <= min_r_squared)])
        excluded_amp = len(df[(df['fit_success']) & (df['amplitude'] <= min_amplitude)])
        print(f"\n📊 Plotting {len(df_success)} pores (R² > {min_r_squared}, amplitude > {min_amplitude})")
        print(f"   Excluded: {excluded_r2} low R², {excluded_amp} low amplitude")
        
        # Adjust figure layout based on whether we have masks
        if mask_info is not None and mask_info['n_masks'] >= 1:
            # Add extra row for per-mask summary (even for single mask)
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        else:
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. FWHM distribution (max) - in nm
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(df_success['fwhm_max_nm'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(df_success['fwhm_max_nm'].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f"Mean: {df_success['fwhm_max_nm'].mean():.1f} nm")
        ax1.set_xlabel('FWHM Max (nm)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('FWHM Distribution (Maximum of X or Y)')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. FWHM X vs Y scatter - in nm (colorblind-friendly)
        ax2 = fig.add_subplot(gs[0, 1])
        scatter = ax2.scatter(df_success['fwhm_x_nm'], df_success['fwhm_y_nm'],
                            c=df_success['r_squared'], cmap='viridis', alpha=0.6, s=20)
        ax2.plot([df_success['fwhm_x_nm'].min(), df_success['fwhm_x_nm'].max()],
                [df_success['fwhm_x_nm'].min(), df_success['fwhm_x_nm'].max()],
                'm--', linewidth=1, label='x=y')  # magenta instead of red
        ax2.set_xlabel('FWHM X (nm)')
        ax2.set_ylabel('FWHM Y (nm)')
        ax2.set_title('FWHM X vs Y (colored by R²)')
        ax2.legend()
        ax2.grid(alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('R²')
        
        # 3. R² distribution (colorblind-friendly green)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(df_success['r_squared'], bins=30, edgecolor='black', alpha=0.7, color='#2ca02c')  # distinct green
        ax3.axvline(df_success['r_squared'].mean(), color='magenta',
                   linestyle='--', linewidth=2, label=f"Mean: {df_success['r_squared'].mean():.3f}")
        ax3.set_xlabel('R² (Goodness of Fit)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Fit Quality Distribution')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. Amplitude distribution (colorblind-friendly)
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.hist(df_success['amplitude'], bins=30, edgecolor='black', alpha=0.7, color='#ff7f0e')  # distinct orange
        ax4.axvline(df_success['amplitude'].mean(), color='magenta',
                   linestyle='--', linewidth=2, label=f"Mean: {df_success['amplitude'].mean():.1f}")
        ax4.set_xlabel('Amplitude')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Peak Intensity Distribution')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        # 5. Sigma X vs Sigma Y (in nm, colorblind-friendly)
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.scatter(df_success['sigma_x_nm'], df_success['sigma_y_nm'], 
                   alpha=0.5, s=20, color='#2ca02c')  # green
        ax5.plot([df_success['sigma_x_nm'].min(), df_success['sigma_x_nm'].max()],
                [df_success['sigma_x_nm'].min(), df_success['sigma_x_nm'].max()],
                'm--', linewidth=1, label='x=y')  # magenta
        ax5.set_xlabel('Sigma X (nm)')
        ax5.set_ylabel('Sigma Y (nm)')
        ax5.set_title('Gaussian Width (Sigma X vs Y)')
        ax5.legend()
        ax5.grid(alpha=0.3)
        
        # 6. FWHM spatial map (using max, magma colormap)
        ax6 = fig.add_subplot(gs[1, 1])
        scatter = ax6.scatter(df_success['fitted_x'], df_success['fitted_y'],
                            c=df_success['fwhm_max_nm'], cmap='magma',
                            s=30, alpha=0.7)
        ax6.set_xlabel('X Position')
        ax6.set_ylabel('Y Position')
        ax6.set_title('FWHM Max Spatial Distribution')
        ax6.invert_yaxis()
        cbar = plt.colorbar(scatter, ax=ax6)
        cbar.set_label('FWHM Max (nm)')
        
        # 7. Orientation distribution
        ax7 = fig.add_subplot(gs[1, 2], projection='polar')
        theta_valid = df_success['theta_rad'].dropna()
        ax7.hist(theta_valid, bins=36, edgecolor='black', alpha=0.7)
        ax7.set_title('Orientation Distribution', y=1.08)
        
        # 8. FWHM vs Amplitude (using max)
        ax8 = fig.add_subplot(gs[1, 3])
        ax8.scatter(df_success['amplitude'], df_success['fwhm_max_nm'],
                   c=df_success['r_squared'], cmap='viridis', alpha=0.6, s=20)
        ax8.set_xlabel('Amplitude')
        ax8.set_ylabel('FWHM Max (nm)')
        ax8.set_title('Amplitude vs FWHM Max')
        ax8.grid(alpha=0.3)
        
        # 9-12. Example fits (best 4)
        best_fits = df_success.nlargest(4, 'r_squared')
        
        for idx, (i, row) in enumerate(best_fits.iterrows()):
            ax = fig.add_subplot(gs[2, idx])
            
            # Find the corresponding fit result
            fit_result = [r for r in fit_results if r['pore_id'] == row['pore_id']][0]
            
            if fit_result['success']:
                # Show original and fitted data (using magma colormap)
                extent = fit_result['window_coords']
                
                ax.imshow(fit_result['window'], cmap='magma', 
                         extent=[extent[2], extent[3], extent[1], extent[0]])
                ax.contour(fit_result['fitted_window'], colors='cyan', alpha=0.6, linewidths=1.5,
                          extent=[extent[2], extent[3], extent[1], extent[0]])
                
                ax.set_title(f"Pore {row['pore_id']}\nFWHM: {row['fwhm_mean_pixels']:.2f}px, R²: {row['r_squared']:.3f}",
                           fontsize=9)
                ax.axis('off')
        
        # Add per-mask summary plots if masks used (even single mask)
        if mask_info is not None and mask_info['n_masks'] >= 1:
            # Plot 1: Pore count and density per mask
            ax_mask1 = fig.add_subplot(gs[3, 0:2])
            mask_counts = []
            mask_densities = []
            mask_labels = []
            for mask_id in range(1, mask_info['n_masks'] + 1):
                mask_data = df_success[df_success['mask_id'] == mask_id]
                count = len(mask_data)
                area_um2 = mask_info['areas_um2'][mask_id - 1]
                density = count / area_um2
                mask_counts.append(count)
                mask_densities.append(density)
                mask_labels.append(f"Mask {mask_id}")
            
            # Bar plot for counts
            x_pos = np.arange(len(mask_labels))
            bars = ax_mask1.bar(x_pos, mask_counts, color='#2ca02c', alpha=0.7, edgecolor='black')
            ax_mask1.set_xticks(x_pos)
            ax_mask1.set_xticklabels(mask_labels)
            ax_mask1.set_ylabel('Pore Count', color='#2ca02c', fontweight='bold')
            ax_mask1.tick_params(axis='y', labelcolor='#2ca02c')
            ax_mask1.set_title('Pore Count and Density per Mask/Cell (R² > 0.7)')
            ax_mask1.grid(alpha=0.3, axis='y')
            
            # Add count labels on bars
            for bar, count, density in zip(bars, mask_counts, mask_densities):
                height = bar.get_height()
                ax_mask1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(count)}\n({density:.2f}/µm²)',
                            ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            # Secondary y-axis for density
            ax_mask1_2 = ax_mask1.twinx()
            ax_mask1_2.plot(x_pos, mask_densities, 'mo-', linewidth=2, markersize=8, label='Density')
            ax_mask1_2.set_ylabel('Pore Density (pores/µm²)', color='magenta', fontweight='bold')
            ax_mask1_2.tick_params(axis='y', labelcolor='magenta')
            
            # Plot 2: FWHM Max distribution per mask (box plot with median and IQR)
            ax_mask2 = fig.add_subplot(gs[3, 2:4])
            mask_fwhm_data = []
            mask_labels_plot = []
            for mask_id in range(1, mask_info['n_masks'] + 1):
                mask_data = df_success[df_success['mask_id'] == mask_id]['fwhm_max_nm']
                if len(mask_data) > 0:
                    mask_fwhm_data.append(mask_data.values)
                    mask_labels_plot.append(f"Mask {mask_id}")
            
            if len(mask_fwhm_data) > 0:
                bp = ax_mask2.boxplot(mask_fwhm_data, labels=mask_labels_plot, 
                                     patch_artist=True, showmeans=True,
                                     medianprops=dict(color='magenta', linewidth=2),
                                     meanprops=dict(marker='D', markerfacecolor='#2ca02c', 
                                                   markeredgecolor='black', markersize=6))
                
                # Color boxes
                for patch in bp['boxes']:
                    patch.set_facecolor('#2ca02c')
                    patch.set_alpha(0.5)
                
                ax_mask2.set_ylabel('FWHM Max (nm)')
                ax_mask2.set_title('FWHM Max Distribution per Mask/Cell\n(Magenta line = median, Green diamond = mean)')
                ax_mask2.grid(alpha=0.3, axis='y')
                
                # Add median and IQR values as text
                for i, (mask_id, data) in enumerate(zip(range(1, mask_info['n_masks'] + 1), mask_fwhm_data)):
                    median = np.median(data)
                    q1 = np.percentile(data, 25)
                    q3 = np.percentile(data, 75)
                    iqr = q3 - q1
                    ax_mask2.text(i + 1, ax_mask2.get_ylim()[0], 
                                f'Med: {median:.1f}\nIQR: {iqr:.1f}',
                                ha='center', va='top', fontsize=8, 
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        fig.suptitle('Nuclear Pore Gaussian Fitting Analysis', fontsize=16, fontweight='bold', y=0.995)
        
        return fig


def create_analysis_widget(viewer, analyzer):
    """Create magicgui widget for analysis controls"""
    
    @magicgui(
        call_button="Run Analysis",
        layout='vertical',
        # Background Subtraction
        bg_subtract_method={
            'widget_type': 'ComboBox',
            'choices': ['none', 'tophat', 'rolling_ball'],
            'value': 'none',
            'label': '🔧 BG Subtract',
            'tooltip': 'Background subtraction method'
        },
        tophat_size={
            'widget_type': 'Slider',
            'min': 5, 'max': 50, 'value': 15,
            'label': 'Top-hat Size',
            'tooltip': 'Disk size for morphological top-hat'
        },
        rolling_ball_radius={
            'widget_type': 'Slider',
            'min': 10, 'max': 200, 'value': 50,
            'label': 'Rolling Ball Radius',
            'tooltip': 'Radius for rolling ball background'
        },
        # Detection Method
        detection_method={
            'widget_type': 'ComboBox',
            'choices': ['log', 'dog', 'doh', 'threshold'],
            'label': '🔴 Detection Method',
            'tooltip': 'Pore detection algorithm'
        },
        # Blob detection parameters
        min_sigma={
            'widget_type': 'FloatSlider',
            'min': 0.5, 'max': 10.0, 'value': 0.50,
            'label': 'Min Sigma',
            'tooltip': 'Minimum sigma for blob detection'
        },
        max_sigma={
            'widget_type': 'FloatSlider',
            'min': 1.0, 'max': 20.0, 'value': 5.0,
            'label': 'Max Sigma',
            'tooltip': 'Maximum sigma for blob detection'
        },
        blob_threshold={
            'widget_type': 'FloatSlider',
            'min': 0.01, 'max': 1.0, 'value': 0.01,
            'label': 'Blob Threshold',
            'tooltip': 'Threshold for blob detection (higher = less sensitive)'
        },
        # Threshold-based detection
        threshold_method={
            'widget_type': 'ComboBox',
            'choices': ['otsu', 'li', 'triangle', 'yen', 'isodata', 'mean', 'minimum', 'manual'],
            'label': 'Threshold Method',
            'tooltip': 'For threshold-based detection'
        },
        manual_threshold={
            'widget_type': 'Slider',
            'min': 0, 'max': 255, 'value': 100,
            'label': 'Manual Threshold',
            'tooltip': 'Manual threshold value'
        },
        smoothing={
            'widget_type': 'FloatSlider',
            'min': 0.0, 'max': 5.0, 'value': 0.5,
            'label': 'Smoothing',
            'tooltip': 'Gaussian smoothing before detection'
        },
        # Gaussian fitting
        fit_window_size={
            'widget_type': 'Slider',
            'min': 9, 'max': 31, 'value': 9,
            'label': '🔬 Fit Window Size',
            'tooltip': 'Window size for Gaussian fitting (odd number)'
        },
        pixel_size_nm={
            'widget_type': 'FloatSpinBox',
            'min': 1.0, 'max': 500.0, 'value': 23.40, 'step': 0.1,
            'label': '📏 Pixel Size (nm)',
            'tooltip': 'Physical size of one pixel in nanometers for unit conversion'
        },
        # Quality filters
        min_r_squared={
            'widget_type': 'FloatSlider',
            'min': 0.0, 'max': 1.0, 'value': 0.0, 'step': 0.05,
            'label': '✓ Min R²',
            'tooltip': 'Minimum R² for including pores in analysis (goodness of fit)'
        },
        min_amplitude={
            'widget_type': 'Slider',
            'min': 0, 'max': 255, 'value': 50,
            'label': '✓ Min Amplitude',
            'tooltip': 'Minimum peak amplitude for confident detections'
        },
        # Output options
        save_results={'widget_type': 'CheckBox', 'value': True, 'label': '💾 Save CSV'},
        show_plots={'widget_type': 'CheckBox', 'value': True, 'label': '📊 Show Plots'},
    )
    def analyze(
        bg_subtract_method: str,
        tophat_size: int,
        rolling_ball_radius: int,
        detection_method: str,
        min_sigma: float,
        max_sigma: float,
        blob_threshold: float,
        threshold_method: str,
        manual_threshold: int,
        smoothing: float,
        fit_window_size: int,
        pixel_size_nm: float,
        min_r_squared: float,
        min_amplitude: int,
        save_results: bool,
        show_plots: bool,
    ):
        """Run nuclear pore Gaussian fitting analysis"""
        
        # Get pore image layer
        try:
            pore_layer = [l for l in viewer.layers if isinstance(l, napari.layers.Image)][0]
        except IndexError:
            print("✗ Could not find image layer")
            return
        
        # Prepare parameters
        bg_params = {
            'tophat_size': tophat_size,
            'rolling_ball_radius': rolling_ball_radius
        }
        
        detection_params = {
            'min_sigma': min_sigma,
            'max_sigma': max_sigma,
            'num_sigma': 5,
            'threshold': blob_threshold,
            'overlap': 0.5,
            'threshold_method': threshold_method,
            'manual_threshold': manual_threshold,
            'min_size': 3,
            'max_size': 200,
            'smoothing': smoothing
        }
        
        # Ensure window size is odd
        if fit_window_size % 2 == 0:
            fit_window_size += 1
        
        # Run analysis
        df, fit_results, mask_info = analyzer.analyze_pores(
            pore_layer.data,
            bg_subtract_method=bg_subtract_method,
            bg_subtract_params=bg_params,
            detection_method=detection_method,
            detection_params=detection_params,
            fit_window_size=fit_window_size,
            pixel_size_nm=pixel_size_nm,
            min_r_squared=min_r_squared,
            min_amplitude=min_amplitude
        )
        
        if len(df) == 0:
            print("✗ No results to save or plot")
            return
        
        # Get image name for output files
        image_name = pore_layer.name.replace(' ', '_').replace('.', '_')
        
        # Save results
        if save_results:
            try:
                csv_filename = f'{image_name}_gaussian_fits.csv'
                df.to_csv(csv_filename, index=False)
                print(f"\n✓ Results saved to '{csv_filename}'")
                print(f"   {len(df)} pores analyzed")
                print(f"   {df['fit_success'].sum()} successful fits")
                
                # Save per-mask summary if any masks used
                if mask_info is not None and mask_info['n_masks'] >= 1:
                    df_high_quality = df[(df['fit_success']) & 
                                        (df['r_squared'] > min_r_squared) & 
                                        (df['amplitude'] > min_amplitude)]
                    summary_data = []
                    for mask_id in range(1, mask_info['n_masks'] + 1):
                        mask_data = df_high_quality[df_high_quality['mask_id'] == mask_id]
                        if len(mask_data) > 0:
                            area_um2 = mask_info['areas_um2'][mask_id - 1]
                            area_pixels = mask_info['areas_pixels'][mask_id - 1]
                            pore_count = len(mask_data)
                            pore_density = pore_count / area_um2
                            
                            summary_data.append({
                                'mask_id': mask_id,
                                'area_pixels': area_pixels,
                                'area_um2': area_um2,
                                'pore_count': pore_count,
                                'pore_density_per_um2': pore_density,
                                'fwhm_max_median_nm': mask_data['fwhm_max_nm'].median(),
                                'fwhm_max_mean_nm': mask_data['fwhm_max_nm'].mean(),
                                'fwhm_max_std_nm': mask_data['fwhm_max_nm'].std(),
                                'fwhm_max_q1_nm': mask_data['fwhm_max_nm'].quantile(0.25),
                                'fwhm_max_q3_nm': mask_data['fwhm_max_nm'].quantile(0.75),
                                'fwhm_max_iqr_nm': mask_data['fwhm_max_nm'].quantile(0.75) - mask_data['fwhm_max_nm'].quantile(0.25),
                                'mean_r_squared': mask_data['r_squared'].mean()
                            })
                    
                    summary_df = pd.DataFrame(summary_data)
                    summary_filename = f'{image_name}_per_mask_summary.csv'
                    summary_df.to_csv(summary_filename, index=False)
                    print(f"✓ Per-mask summary saved to '{summary_filename}'")
                    
            except Exception as e:
                print(f"✗ Could not save results: {e}")
        
        # Show plots
        if show_plots:
            try:
                fig = analyzer.plot_results(df, fit_results, pore_layer.data, mask_info, 
                                          min_r_squared, min_amplitude)
                if fig is not None:
                    # Auto-save high-res PNG
                    plot_filename = f'{image_name}_analysis_plots.png'
                    fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
                    print(f"✓ Plots saved to '{plot_filename}' (300 DPI)")
                    plt.show(block=False)
            except Exception as e:
                print(f"✗ Could not create plots: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n✓ Analysis complete!")
        return df
    
    return analyze


if __name__ == "__main__":
    # Initialize napari viewer
    viewer = napari.Viewer()
    
    # Generate synthetic pore data (reduced to 50 pores to prevent crashes)
    print("Generating synthetic nuclear pore data...")
    pores = generate_synthetic_pore_data(n_pores=50)
    
    # Add to viewer
    viewer.add_image(pores, name='Nuclear Pores', colormap='magma', blending='additive')
    
    # Add shapes layer for ROI selection
    shapes_layer = viewer.add_shapes(
        name='ROI Mask',
        shape_type='polygon',
        edge_width=2,
        edge_color='cyan',
        face_color=[0, 1, 1, 0.2]  # transparent cyan
    )
    
    # Create analyzer and widget
    analyzer = NuclearPoreGaussianAnalyzer(viewer)
    widget = create_analysis_widget(viewer, analyzer)
    viewer.window.add_dock_widget(widget, area='right', name='Gaussian Fit Analysis')
    
    print("\n" + "="*60)
    print("NUCLEAR PORE GAUSSIAN FITTING ANALYSIS WITH ROI MASKING")
    print("="*60)
    print("\nFeatures:")
    print("1. ROI Selection:")
    print("   - Draw shapes on 'ROI Mask' layer to select regions")
    print("   - Use polygon, rectangle, or ellipse tools")
    print("   - Analysis will only count pores inside drawn shapes")
    print("   - Leave empty to analyze all pores")
    print("\n2. Background Subtraction:")
    print("   - None, Top-hat, Rolling Ball")
    print("\n3. Pore Detection:")
    print("   - LoG/DoG/DoH blob detection")
    print("   - Threshold-based segmentation")
    print("\n4. Gaussian Fitting:")
    print("   - 2D Gaussian fit to each pore")
    print("   - FWHM calculation (X, Y, mean)")
    print("   - Fit quality (R²)")
    print("   - Orientation and amplitude")
    print("\n4. Comprehensive Visualization:")
    print("   - FWHM distributions")
    print("   - Spatial maps")
    print("   - Example fits")
    print("   - Quality metrics")
    print("\n5. Detailed CSV output with all parameters")
    print("="*60 + "\n")
    
    napari.run()
