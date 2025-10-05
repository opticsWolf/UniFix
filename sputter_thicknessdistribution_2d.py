import numpy as np
from numba import njit, prange
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from scipy.ndimage import gaussian_filter

#"old" functin for thickness along line calculation
@njit(parallel=True, fastmath=True, cache=True)
def _flux_numba(tx: np.ndarray, ty: np.ndarray,
                src_x: np.ndarray, src_y: np.ndarray,
                d2: float, dA: float, cos_theta0: float, sigma: float) -> np.ndarray:
    """
    Compute flux contributions for many target points and a single source rectangle.
    Optimized to use vectorization and parallel processing where possible.

    Parameters
    ----------
    tx : ndarray
        x coordinates of the target points.
    ty : ndarray
        y coordinates of the target points.
    src_x : ndarray
        x coordinates of the source rectangle vertices.
    src_y : ndarray
        y coordinates of the source rectangle vertices.
    d2 : float
        Squared vertical distance from sources to deposition plane (z^2).
    dA : float
        Area element for the source rectangle.
    cos_theta0 : float
        Preferred emission direction cosine with respect to surface normal.
    sigma : float
        Angular spread parameter.

    Returns
    -------
    ndarray
        Flux contributions for each target point.
    """
    n_points = tx.shape[0]
    flux = np.zeros(n_points, dtype=np.float64)
    eps = np.finfo(np.float64).eps

    for i in prange(n_points):
        total = 0.0
        for j in range(src_x.shape[0]):
            dx = tx[i] - src_x[j]
            dy = ty[i] - src_y[j]
            r_sq = dx*dx + dy*dy + d2

            # Avoid division by zero and square root of negative numbers
            cos_theta = np.sqrt(d2 / max(r_sq, d2 + eps))
            weight = np.exp(-((cos_theta - cos_theta0)**2) / (2.0 * sigma**2 + eps))

            total += (1.0 / (r_sq + eps)) * weight * dA
        flux[i] = total

    return flux

@njit(parallel=True, fastmath=True, cache=True)
def _calculate_flux_analytical_numba(x_grid: np.ndarray, y_grid: np.ndarray,
                               source_points_x: np.ndarray, source_points_y: np.ndarray,
                               x_source_positions: np.ndarray, shift_y: np.ndarray,
                               d2: float, cos_theta0: float, sigma: float) -> np.ndarray:
    """
    Calculate flux distribution analytically with shifted source positions using Numba acceleration.
    This is an internal implementation optimized for performance.

    Parameters
    ----------
    x_grid : np.ndarray
        2D grid of x-coordinates (shape: (nx, ny)).
    y_grid : np.ndarray
        2D grid of y-coordinates (same shape as x_grid).
    source_points_x : np.ndarray
        1D array of source x-coordinates (shape: (n_src,)).
    source_points_y : np.ndarray
        1D array of source y-coordinates (shape: (n_src,), matching source_points_x).
    x_source_positions : np.ndarray
        1D array of x-shift positions for sources (shape: (n_shifts,)).
    shift_y : np.ndarray
        1D array of y-shifts corresponding to each x_source position (same length as x_source_positions).
    d2 : float
        Squared distance parameter in the calculation.
    cos_theta0 : float
        Reference cosine angle for Gaussian weighting function.
    sigma : float
        Standard deviation for the Gaussian angular weight.

    Returns
    -------
    np.ndarray
        2D array of same shape as input grids containing calculated flux values at each grid point.

    Notes
    -----
    - Uses Numba with parallel=True, fastmath=True, cache=True for optimized performance.
    - The calculation involves summing contributions from all shifted source points,
      where each contribution is weighted by a Gaussian function in angular space.
    - Small epsilon is added to prevent division by zero and numerical errors.
    """
    n_src = source_points_x.shape[0]
    nx, ny = x_grid.shape
    thickness_map = np.zeros((nx, ny), dtype=np.float64)
    eps = np.finfo(np.float64).eps

    for i in prange(nx):
        for j in prange(ny):
            total_contrib = 0.0
            for k in range(len(x_source_positions)):
                for l in range(n_src):
                    dx = x_grid[i, j] - (x_source_positions[k] + source_points_x[l])
                    dy = y_grid[i, j] - (source_points_y[l] + shift_y[k])
                    r_sq = dx**2 + dy**2 + d2

                    # Avoid division by zero and square root of negative numbers
                    cos_theta = np.sqrt(d2 / max(r_sq, d2 + eps))
                    weight_angular = np.exp(-((cos_theta - cos_theta0)**2) / (2.0 * sigma**2 + eps))

                    contrib = 1.0 / (r_sq + eps) * weight_angular
                    total_contrib += contrib

            thickness_map[i, j] = total_contrib

    return thickness_map

@njit(parallel=True, fastmath=True)
def _compute_thickness_along_line_numba(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    thickness_map: np.ndarray,
    radii: np.ndarray,
    n_phi: int
) -> np.ndarray:
    """
    Compute azimuthally averaged thickness along radial positions
    using direct bilinear interpolation on a regular grid.
    Masked regions (zeros) contribute zero deposition.

    Parameters
    ----------
    x_grid, y_grid : 2D arrays
        Cartesian coordinate grids.
    thickness_map : 2D array
        Deposition thickness (masked regions are zero).
    radii : 1D array
        Radial positions to evaluate.
    n_phi : int
        Number of azimuthal samples.

    Returns
    -------
    thickness_along_line : 1D ndarray
        Averaged thickness at each radius.
    """
    ny, nx = x_grid.shape
    x_min, x_max = x_grid[0, 0], x_grid[0, -1]
    y_min, y_max = y_grid[0, 0], y_grid[-1, 0]
    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)

    phis = np.linspace(-np.pi, np.pi, n_phi)
    tline = np.zeros(radii.size)

    for ir in prange(radii.size):
        r = radii[ir]
        acc = 0.0
        for phi in phis:
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            # Convert to fractional indices
            ix = (x - x_min) / dx
            iy = (y - y_min) / dy
            if ix < 0 or iy < 0 or ix >= nx - 1 or iy >= ny - 1:
                continue
            i0 = int(ix)
            j0 = int(iy)
            i1 = i0 + 1
            j1 = j0 + 1
            wx = ix - i0
            wy = iy - j0
            # Bilinear interpolation
            v00 = thickness_map[j0, i0]
            v10 = thickness_map[j0, i1]
            v01 = thickness_map[j1, i0]
            v11 = thickness_map[j1, i1]
            val = (
                v00 * (1 - wx) * (1 - wy)
                + v10 * wx * (1 - wy)
                + v01 * (1 - wx) * wy
                + v11 * wx * wy
            )
            acc += val
        tline[ir] = acc / n_phi
    return tline


@njit(parallel=True, fastmath=True, cache=True)
def _generate_shadow_mask_edge_numba(x_grid: np.ndarray, y_grid: np.ndarray, x_grow_cols: np.ndarray) -> np.ndarray:
    """
    Generate a shadow mask that grows inward from left/right edges along x for each y row.
    The edge growth is smoothed using a natural cubic spline across rows.

    Parameters
    ----------
    x_grid : 2D ndarray (ny, nx)
    y_grid : 2D ndarray (ny, nx)
    x_grow_cols : 1D ndarray
        Number of x-columns to mask from left/right edges per row (length ny)

    Returns
    -------
    mask : 2D ndarray of bool
        True = blocked, False = open
    """
    # --- Cubic spline smoothing ---
    ny = len(x_grow_cols)
    x = np.arange(ny)
    n = ny
    h = np.zeros(n - 1)
    alpha = np.zeros(n - 1)
    
    for i in range(n - 1):
        h[i] = x[i + 1] - x[i]
    for i in range(1, n - 1):
        alpha[i] = (3 / h[i]) * (x_grow_cols[i + 1] - x_grow_cols[i]) - (3 / h[i - 1]) * (x_grow_cols[i] - x_grow_cols[i - 1])
    
    l = np.ones(n)
    mu = np.zeros(n)
    z = np.zeros(n)
    
    for i in range(1, n - 1):
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
    
    b = np.zeros(n - 1)
    c = np.zeros(n)
    d = np.zeros(n - 1)
    
    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (x_grow_cols[j + 1] - x_grow_cols[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])
    
    # Evaluate spline at original points
    x_grow_cols_smooth = np.zeros(n)
    for i in range(n - 1):
        dx = 0  # evaluate at x[i]
        x_grow_cols_smooth[i] = x_grow_cols[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
    x_grow_cols_smooth[-1] = x_grow_cols[-1]
    
    # --- Generate shadow mask using smoothed grow values ---
    ny, nx = x_grid.shape
    mask = np.zeros((ny, nx), dtype=np.bool_)
    
    for i in prange(ny):
        grow = int(min(max(x_grow_cols_smooth[i], 0), nx // 2))
        if grow > 0:
            # mask from left
            for j in range(grow):
                mask[i, j] = True
            # mask from right
            for j in range(nx - grow, nx):
                mask[i, j] = True
    
    return mask


class ThicknessDistributionCalculator:
    """
    A class to calculate thickness distribution across a rectangular opening with line or area sources.
    Provides both analytical and numerical calculation methods, with support for angular spread.
    """

    def __init__(
        self,
        inner_radius: float,
        outer_radius: float,
        opening_width_cm: float,
        source_distance_from_center_cm: float,
        distance_from_deposition_plane_cm: float,
        grid_resolution: int = 100,
        source_length_cm: Optional[float] = None,
        num_sources: int = 1,
        source_spacing_cm: Optional[float] = None,
        source_width_cm: float = 5.0,
        cos_theta0: float = 1.0,       # Default: perfectly directed emission
        sigma: float = 0.1,             # Default narrow angular spread
        normalize: bool = True,
        shift_y_cm: Optional[float] = None
    ):
        """
        Initialize the ThicknessDistributionCalculator with the given parameters.
        Validates input parameters to ensure robustness.

        Parameters
        ----------
        inner_radius : float
            Inner radius of the turntable in cm.
        outer_radius : float
            Outer radius of the turntable in cm.
        opening_width_cm : float
            Width of the sputter station opening in cm.
        source_distance_from_center_cm : float
            Horizontal distance from the center where sources are located.
        distance_from_deposition_plane_cm : float
            Vertical distance from sources to deposition plane (z).
        grid_resolution : int, optional
            Resolution of the deposition grid.
        source_length_cm : float or None, optional
            Length of the sources in cm. If None, spans full opening.
        num_sources : int, optional
            Number of rectangular sources to use (default is 1).
        source_spacing_cm : float or None, optional
            Explicit spacing between sources (None for automatic calculation).
        source_width_cm : float, optional
            Width of each rectangular source in cm.
        cos_theta0 : float, optional
            Preferred emission direction cosine with respect to surface normal (1.0 for perfectly directed).
        sigma : float, optional
            Angular spread parameter.
        normalize : bool, optional
            Whether to normalize the thickness map so that its maximum is 1.0.
        shift_y_cm : float or None, optional
            Shift of the sources along y-axis relative to center. If not None,
            sources will be shifted by -shift_y_cm for first source, +shift_y_cm for last source,
            and evenly spaced in between.
        """
        if inner_radius < 0 or outer_radius <= inner_radius:
            raise ValueError("Invalid radii: inner_radius must be positive and less than outer_radius.")
        if opening_width_cm <= 0:
            raise ValueError("opening_width_cm must be positive.")
        if distance_from_deposition_plane_cm < 0:
            raise ValueError("distance_from_deposition_plane_cm must be non-negative.")

        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.opening_width_cm = opening_width_cm
        self.source_distance_from_center_cm = source_distance_from_center_cm
        self.distance_from_deposition_plane_cm = distance_from_deposition_plane_cm
        self.grid_resolution = grid_resolution
        self.source_length_cm = source_length_cm if source_length_cm is not None else outer_radius - inner_radius
        self.num_sources = num_sources
        self.source_spacing_cm = source_spacing_cm
        self.source_width_cm = source_width_cm
        self.cos_theta0 = cos_theta0
        self.sigma = sigma
        self.normalize = normalize
        self.shift_y_cm = shift_y_cm

        # Precompute constants to avoid repeated calculations
        #self.grid_factor = (self.outer_radius - self.inner_radius) / self.opening_width_cm
        self.grid_factor = 1
        
        self.d2 = distance_from_deposition_plane_cm**2
        self.eps = np.finfo(float).eps

        #self.x_grid = None
        #self.y_grid = None
        #self.thickness_map = None
        self.source_x_positions = []
        self.shift_y_positions = []

    def _angular_weight(self, r_sq: np.ndarray) -> np.ndarray:
        """
        Calculate angular weight for sputter deposition.

        Parameters
        ----------
        r_sq : ndarray
            Squared total distance from source to deposition point (x^2 + y^2 + z^2).

        Returns
        -------
        ndarray
            Angular weight, normalized so that max is 1.
        """
        # Avoid division by zero and square root of negative numbers
        r_sq_clamped = np.maximum(r_sq, self.d2 + self.eps)

        # Calculate cos(theta) between emission direction and target normal
        cos_theta = np.sqrt(self.d2 / r_sq_clamped)

        # Gaussian-like weighting in cosine space
        weight = np.exp(-((cos_theta - self.cos_theta0)**2) / (2.0 * self.sigma**2 + self.eps))

        return weight


    def _get_source_x_positions(self) -> List[float]:
        """
        Calculate the x positions of the sources based on the given parameters.

        Returns
        -------
        list[float]
            The x positions of the sources.
        """
        if self.num_sources > 1:
            if self.source_spacing_cm is not None and self.source_spacing_cm <= 0:
                raise ValueError("source_spacing_cm must be positive when provided")

            if self.source_spacing_cm is not None:
                # Explicit spacing: symmetric array centered at 0, then shifted
                total_span = (self.num_sources - 1) * self.source_spacing_cm
                start_pos = -total_span / 2.0
                source_x_positions = [start_pos + i * self.source_spacing_cm for i in range(self.num_sources)]
            else:
                # Automatic spacing: fill full opening width
                total_span = self.opening_width_cm
                source_spacing = total_span / (self.num_sources + 1)
                start_pos = -self.opening_width_cm / 2 + source_spacing
                source_x_positions = [start_pos + i * source_spacing for i in range(self.num_sources)]

            # Apply global shift to the center position
            return [x + self.source_distance_from_center_cm for x in source_x_positions]
        else:
            # For a single source, just use the given distance
            return [self.source_distance_from_center_cm]

    def _get_shift_y_positions(self) -> List[float]:
        """
        Calculate shift positions along y-axis based on the given parameters.

        Returns
        -------
        list[float]
            The y shifts of the sources.
        """
        if self.shift_y_cm is None:
            return [0.0] * self.num_sources

        # Calculate evenly spaced shifts between -shift_y_cm and +shift_y_cm
        step = (2 * self.shift_y_cm) / (self.num_sources - 1) if self.num_sources > 1 else 0.0
        return [-self.shift_y_cm + i * step for i in range(self.num_sources)]

    def calculate_thickness_distribution(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate thickness distribution using the specified method with angular spread.

        Returns
        -------
        tuple[ndarray, ndarray, ndarray]
            (x_grid, y_grid, thickness_map)
        """
        # Create grids for the deposition plane
        
        x = np.linspace(-self.opening_width_cm / 2, self.opening_width_cm / 2, self.grid_resolution)
        y = np.linspace(self.inner_radius, self.outer_radius, int(self.grid_resolution * self.grid_factor))
        x_grid, y_grid = np.meshgrid(x, y)

        # Calculate source positions
        self.source_x_positions = self._get_source_x_positions()
        self.shift_y_positions = self._get_shift_y_positions()

        if self.source_length_cm is None:
            self.source_length_cm = self.outer_radius - self.inner_radius

        # Sample uniformly across area of the target surface(s)
        xs = np.linspace(-self.source_width_cm / 2, self.source_width_cm / 2, self.grid_resolution)
        ys = np.linspace(0.5 * (y[0] + y[-1]) - self.source_length_cm / 2,
                         0.5 * (y[0] + y[-1]) + self.source_length_cm / 2,
                         int(self.grid_resolution*self.grid_factor))
        source_points_x, source_points_y = np.meshgrid(xs, ys)

        # Use Numba-accelerated version for better performance
        thickness_map = _calculate_flux_analytical_numba(
            x_grid, y_grid,
            source_points_x.ravel(), source_points_y.ravel(),
            np.array(self.source_x_positions),
            np.array(self.shift_y_positions),
            self.d2,
            self.cos_theta0,
            self.sigma
        )

        if self.normalize and thickness_map.max() > 0:  # Avoid division by zero
            thickness_map = (thickness_map / thickness_map.max())
        elif self.normalize:
            thickness_map[:] = 0.5

        self.x_grid = x_grid
        self.y_grid = y_grid
        self.thickness_map = thickness_map

        return x_grid, y_grid, thickness_map
    
    #'old' method for thickness along line but more accurate
    def calculate_thickness_along_line(
        self,
        substrate_inner_radius: float,
        substrate_outer_radius: float,
        n_phi: int = 40000,  # Increased from default 20000 for smoother curve
        deposition_rate_nm_per_s: Optional[float] = None,
        rpm: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate thickness along a radial line with multiple sources using Numba.
        Optimized to vectorize the flux calculation for each source position.
    
        Parameters
        ----------
        substrate_inner_radius : float
            Inner radius of the substrate in cm.
        substrate_outer_radius : float
            Outer radius of the substrate in cm.
        n_phi : int, optional
            Number of points along a circular arc at constant r (default: 40000).
        deposition_rate_nm_per_s : float or None, optional
            If provided, converts thickness to nm/s by scaling appropriately.
        rpm : float or None, optional
            Rotational speed in RPM. Required for conversion if deposition_rate_nm_per_s is provided.
    
        Returns
        -------
        tuple[ndarray, ndarray]
            (radii, thickness_along_line)
        """
        eps = np.finfo(float).eps
        radii = np.linspace(substrate_inner_radius, substrate_outer_radius, int(self.grid_resolution * self.grid_factor))
        phis = np.linspace(0.0, 2.0*np.pi, n_phi, endpoint=False)
    
        # Source rectangle bounds (local), now including opening_width
        x_min = -self.opening_width_cm / 2
        x_max = self.opening_width_cm / 2
        y_center = 0.5 * (self.inner_radius + self.outer_radius)
        y_min = y_center - self.source_length_cm / 2
        y_max = y_center + self.source_length_cm / 2
    
        n_side = int(np.sqrt(self.line_samples))
        xs = np.linspace(x_min, x_max, n_side)
        ys = np.linspace(y_min, y_max, n_side)
        xv, yv = np.meshgrid(xs, ys, indexing="xy")
        base_src_x = xv.ravel()
        base_src_y = yv.ravel()
        dA = (x_max - x_min) / n_side * (y_max - y_min) / n_side
    
        # Get all source positions
        self.source_x_positions = self._get_source_x_positions()
        self.shift_y_positions = self._get_shift_y_positions()
    
        thickness_along_line = np.zeros_like(radii)
    
        for i, r in enumerate(radii):
            tx = r * np.cos(phis)
            ty = r * np.sin(phis)
    
            total_flux = np.zeros_like(tx)
            for k in range(len(self.source_x_positions)):
                src_x = base_src_x + self.source_x_positions[k]
                src_y = base_src_y + self.shift_y_positions[k]
                total_flux += _flux_numba(tx, ty, src_x, src_y, self.d2, dA, self.cos_theta0, self.sigma)
    
            thickness_along_line[i] = np.trapezoid(total_flux, phis) / (2.0 * np.pi)
    
        ## Apply a small amount of smoothing to the curve
        #if n_phi > 1000:  # Only smooth if sufficient samples are available
        #    from scipy.ndimage import gaussian_filter1d
        #    thickness_along_line = gaussian_filter1d(thickness_along_line, sigma=1)
    
        # Convert to nm/s if requested
        if deposition_rate_nm_per_s is not None and rpm is not None and rpm > 0:
            norm_factor = thickness_along_line.mean() + eps
            thickness_along_line = deposition_rate_nm_per_s * (thickness_along_line / norm_factor)
        else:
            if self.normalize and thickness_along_line.max() > 0:
                thickness_along_line /= thickness_along_line.max()
    
        return radii, thickness_along_line
       
    # -------------------------------------------------------------
    # SHADOW MASK GENERATION AND OPTIMIZATION
    # -------------------------------------------------------------
    
    def _half_profile_to_thresholds(self, half_profile: np.ndarray, x_half: np.ndarray) -> np.ndarray:
        """Linear interpolation of control depths across x_half."""
        return np.interp(x_half, np.linspace(0, x_half.max(), len(half_profile)), half_profile)
    
        
    def generate_shadow_mask_columns(self, y_grow_cols: np.ndarray) -> np.ndarray:
        """
        Generate shadow mask using y-column growth.
        
        Parameters
        ----------
        y_grow_cols : 1D ndarray
            Number of y-rows to mask from top and bottom per x-column.
    
        Returns
        -------
        mask : 2D ndarray
            Boolean mask array.
        """
        return _generate_shadow_mask_edge_numba(self.x_grid, self.y_grid, y_grow_cols)

    def _thickness_along_line_from_map_numba(
        self,
        thickness_map_override: np.ndarray,
        substrate_inner_radius: float,
        substrate_outer_radius: float,
        n_phi: int = 40000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute azimuthally averaged thickness using fast Numba routine."""
        x_grid = self.x_grid
        y_grid = self.y_grid
        radii = np.linspace(substrate_inner_radius, substrate_outer_radius, int(self.grid_resolution * self.grid_factor))
        tline = _compute_thickness_along_line_numba(x_grid, y_grid, thickness_map_override, radii, n_phi)
        if np.max(tline) > 0:
            tline /= np.max(tline)
        return radii, tline
      
        
    def optimize_shadow_mask(
        self,
        substrate_inner_radius: float,
        substrate_outer_radius: float,
        n_cols: int = 10,
        n_phi: int = 40000,
        maxiter: int = 60,
        popsize: int = 15,
        optimizer_opts: Optional[dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        if not hasattr(self, "thickness_map"):
            self.calculate_thickness_distribution()
    
        nx = self.x_grid.shape[1]
        x_indices = np.linspace(0, nx - 1, n_cols).astype(int)
    
        bounds = [(0.0, self.opening_width_cm / 2.0) for _ in range(n_cols)]
        print (bounds, bounds)
    
        # Calculate initial thickness profile
        radii, tline_initial = self._thickness_along_line_from_map_numba(
            self.thickness_map, substrate_inner_radius, substrate_outer_radius, n_phi*2
        )
    
        # Create initial guess from tline_initial
        min_t = np.min(tline_initial)
        max_t = np.max(tline_initial)
        if max_t == min_t:
            tline_norm = np.zeros_like(tline_initial)  # avoid division by zero
        else:
            tline_norm = (tline_initial - min_t) / (max_t - min_t)
    
        # Get representative x positions for columns (using middle row of grid if 2D)
        if self.y_grid.ndim == 2:
            y_positions = self.y_grid[:,0]  # Assuming symmetric case
        else:
            y_positions = self.y_grid
    
        # Create initial guess by sampling at column positions directly
        y_column_positions = y_positions[x_indices]
        #scale_factor = (self.y_grid.shape[0] / 2.0)
        scale_factor = np.abs(1.0/max(0.55, min_t)) * self.opening_width_cm/4.0
        initial_y_grow = np.interp(y_column_positions, radii, tline_norm) * scale_factor
        print (y_column_positions)
    
        # Pre-optimization with gradient-based method (L-BFGS-B)
        def smooth_objective_preopt(y_grow_flat: np.ndarray) -> float:
            y_grow_cols = np.interp(np.arange(nx), x_indices, y_grow_flat)
            mask = self.generate_shadow_mask_columns(y_grow_cols)
    
            mask_float = mask.astype(float)
            mask_smoothed = gaussian_filter(mask_float, sigma=2.0)
            tmap_masked = self.thickness_map.copy() * (1 - mask_smoothed)
    
            _, tline = self._thickness_along_line_from_map_numba(
                tmap_masked, substrate_inner_radius, substrate_outer_radius, n_phi
            )
            return float(np.std(tline))
    

        bounds_preopt = [(0.1, self.opening_width_cm/2.1,) for _ in range(n_cols)]
        result_preopt = minimize(
            smooth_objective_preopt,
            initial_y_grow,
            bounds=bounds_preopt,
            method='L-BFGS-B',
            options={'maxiter': 100}
        )
        best_y_grow_initial = result_preopt.x


        def objective(y_grow_flat: np.ndarray) -> float:
            y_grow_cols = np.interp(np.arange(nx), x_indices, y_grow_flat)
            mask = self.generate_shadow_mask_columns(y_grow_cols)
    
            mask_float = mask.astype(float)
            mask_smoothed = gaussian_filter(mask_float, sigma=2.0)
            tmap_masked = self.thickness_map.copy() * (1 - mask_smoothed)
    
            _, tline = self._thickness_along_line_from_map_numba(
                tmap_masked, substrate_inner_radius, substrate_outer_radius, n_phi
            )
            return float(np.std(tline))
    
        de_opts = dict(maxiter=maxiter, popsize=popsize, disp=False)
        if optimizer_opts:
            de_opts.update(optimizer_opts)
    
        result = differential_evolution(
            objective,
            bounds,
            x0=best_y_grow_initial,
            **de_opts
        )
        best_y_grow = result.x

        #best_y_grow = best_y_grow_initial
        best_mask = self.generate_shadow_mask_columns(np.interp(np.arange(nx), x_indices, best_y_grow))
    
        mask_float = best_mask.astype(float)
        mask_smoothed = gaussian_filter(mask_float, sigma=2.0)
        tmap_final = self.thickness_map.copy() * (1 - mask_smoothed)
    
        radii, tline_best = self._thickness_along_line_from_map_numba(
            tmap_final, substrate_inner_radius, substrate_outer_radius, n_phi*2
        )
    
        return best_y_grow, mask_smoothed, (radii, tline_initial), (radii, tline_best)


def plot_and_analyze_thickness(
    inner_radius: float,
    outer_radius: float,
    substrate_inner_radius: float,
    substrate_outer_radius: float,
    opening_width_cm: float,
    source_distance_from_center_cm: float,
    distance_from_deposition_plane_cm: float,
    rpm: float = 150.0,  # Rotation speed for radial motion (not used in this function but included for context)
    deposition_rate_nm_per_s: float = 0.5,
    grid_resolution: int = 250,
    source_length_cm: Optional[float] = None,
    num_sources: int = 1,
    source_spacing_cm: Optional[float] = None,
    source_width_cm: float = 5.0,
    cos_theta0: float = 1.0,
    sigma: float = 0.1,
    normalize: bool = True,
    shift_y_cm: Optional[float] = None
) -> None:
    """
    Calculate, plot the thickness distribution map, and analyze the thickness variation along a radial line.

    Parameters are the same as in calculate_source_thickness_distribution.
    Additionally, rpm is provided for context (not used in this function).
    """
    # Initialize the calculator with all parameters
    calculator = ThicknessDistributionCalculator(
        inner_radius,
        outer_radius,
        opening_width_cm,
        source_distance_from_center_cm,
        distance_from_deposition_plane_cm,
        grid_resolution=grid_resolution,
        source_length_cm=source_length_cm if source_length_cm is not None else (outer_radius - inner_radius),
        num_sources=num_sources,
        source_spacing_cm=source_spacing_cm,
        source_width_cm=source_width_cm,
        cos_theta0=cos_theta0,
        sigma=sigma,
        normalize=normalize,
        shift_y_cm=shift_y_cm
    )
    """
    Runs optimization, plots initial and optimized thickness lines, and shows best mask.
    """
    # Run optimization
    best_ctrl, best_mask, (radii_init, tline_init), (radii_best, tline_best) = calculator.optimize_shadow_mask(
        substrate_inner_radius=substrate_inner_radius,
        substrate_outer_radius=substrate_outer_radius,
        n_cols=20,
        n_phi=80000,
        maxiter=50,
        popsize=15,
    )

    # Get base data for context
    x_grid, y_grid, thickness_map = calculator.x_grid, calculator.y_grid, calculator.thickness_map

    # --- Plot setup ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # 1) Thickness map
    contour = ax1.contourf(x_grid, y_grid, thickness_map, levels=20, cmap="magma")
    contour.set_clim(0.0, 1.0)
    cbar = fig.colorbar(contour, ax=ax1, label="Normalized Thickness")
    cbar.set_ticks(np.linspace(np.min(thickness_map), np.max(thickness_map), 5))
    ax1.set_title(
        f"Initial Thickness Map\n{num_sources} source(s), width={source_width_cm:.1f} cm, "
        f"cos(θ₀)={cos_theta0:.2f}, σ={sigma:.2f}"
    )
    
    for i in range(num_sources):
        rect_x = calculator.source_x_positions[i] - source_width_cm / 2
        rect_y = (inner_radius + outer_radius) / 2 - source_length_cm / 2
        if shift_y_cm is not None:
            rect_y += calculator.shift_y_positions[i]
        ax1.add_patch(plt.Rectangle(
            (rect_x, rect_y),
            source_width_cm,
            source_length_cm,
            linewidth=1.5,
            edgecolor='red',
            facecolor='none',
            linestyle='--'
        ))
    
    ax1.set_xlabel("Width (cm)")
    ax1.set_ylabel("Radius (cm)")

    # 2) Thickness profiles before and after optimization
    ax2.plot(radii_init, tline_init, "b-", linewidth=2, label="Initial")
    ax2.plot(radii_best, tline_best, "r--", linewidth=2, label="Optimized")
    ax2.set_xlabel("Radius (cm)")
    ax2.set_ylabel("Normalized Thickness")
    ax2.set_title(
        f"Radial Thickness Profiles\nOptimized ctrl: {np.round(best_ctrl, 3)}"
    )
    ax2.legend()
    ax2.grid(True, linestyle="--", linewidth=0.5)

    # 3) Best shadow mask visualization
    ax3.imshow(
        best_mask,
        extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
        origin="lower",
        cmap="gray_r",  # white=open, black=masked
        interpolation="none",
    )
    ax3.set_title("Optimized Shadow Mask (grows inward from edges)")

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Default parameters
    inner_radius = 350.0   # cm, half of the inner diameter
    outer_radius = 380.0   # cm, half of the outer diameter
    opening_width_cm = 20.0  # cm
    source_length_cm = 30.0  # cm, None = Use full opening length
    distance_from_deposition_plane_cm = 5.0  # cm
    substrate_inner_radius = inner_radius + 1.0
    substrate_outer_radius = outer_radius - 1.0

    print(f"Example: with directed emission (cos(θ₀)=1, σ=0.5), no y-shift")
    plot_and_analyze_thickness(
        inner_radius,
        outer_radius,
        substrate_inner_radius,
        substrate_outer_radius,
        opening_width_cm,
        source_distance_from_center_cm=0.0,  # centered in the opening
        distance_from_deposition_plane_cm=distance_from_deposition_plane_cm,
        source_length_cm = source_length_cm,
        grid_resolution=100,
        num_sources=2,
        source_spacing_cm=10.0,  # automatic spacing for three sources
        source_width_cm=1.0,
        cos_theta0=1.0,
        sigma=0.5,
        normalize=True,
        shift_y_cm=None
    )
    '''
    print(f"Example: with directed emission (cos(θ₀)=1, σ=0.5), y-shift of 5 cm")
    plot_and_analyze_thickness(
        inner_radius,
        outer_radius,
        substrate_inner_radius,
        substrate_outer_radius,
        opening_width_cm,
        source_distance_from_center_cm=0.0,  # centered in the opening
        distance_from_deposition_plane_cm=distance_from_deposition_plane_cm,
        source_length_cm = 5,
        grid_resolution=100,
        num_sources=3,
        source_spacing_cm=15,  # automatic spacing for three sources
        source_width_cm=1.5,
        cos_theta0=1.5,
        sigma=0.5,
        normalize=True,
        shift_y_cm=10.0
    )
    
    # Create grid and parameters
    x = np.linspace(-15, 15, 100)
    y = np.linspace(0, 25, 100)
    xg, yg = np.meshgrid(x, y)
    ctrl = np.array([1.5, 2.5, 15.5, 2.5, 1.5])  # grows inward
    mask = _generate_shadow_mask_edge_numba(xg, yg, ctrl)
    
    # Plot with verified colormap
    plt.imshow(mask, extent=[x.min(), x.max(), y.min(), y.max()],
               origin='lower', cmap='gray_r')  # gray_r: black=open, white=blocked
    plt.title("Symmetric Shadow Mask (inward growth)")
    plt.xlabel("x [cm]")
    plt.ylabel("y [cm]")
    
    # Add colorbar for verification
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.ax.set_yticklabels(['Open', 'Blocked'])
    
    plt.show()
    '''