"""
Utility classes and functions for spatialize-examples notebooks.

This module provides helper classes for generating synthetic test scenarios
and working with real-world case studies in the spatialize-examples repository.

Classes
-------
SyntheticScenario
    Generate synthetic spatial interpolation scenarios with known ground truth.
PrecipitationCaseStudy
    Real-world precipitation case study with visualization utilities.
SuppressOutput
    Context manager to temporarily suppress stdout output.

Functions
---------
calculate_extent
    Calculate spatial extent from coordinate columns in a DataFrame.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import sys
import os
from pathlib import Path

import import_helper
from spatialize.viz import PlotStyle, plot_colormap_data, plot_nongriddata


# Helper functions for case studies
def calculate_extent(dataframe, x_col='X', y_col='Y'):
    """
    Calculate spatial extent from coordinate columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing spatial coordinates.
    x_col : str, optional
        Name of the X coordinate column. Default is 'X'.
    y_col : str, optional
        Name of the Y coordinate column. Default is 'Y'.

    Returns
    -------
    list
        Extent as [xmin, xmax, ymin, ymax].

    Examples
    --------
    >>> df = pd.DataFrame({'X': [0, 10, 5], 'Y': [0, 20, 10]})
    >>> calculate_extent(df)
    [0, 10, 0, 20]
    """
    return [
        dataframe[x_col].min(),
        dataframe[x_col].max(),
        dataframe[y_col].min(),
        dataframe[y_col].max()
    ]


class SyntheticScenario:
    """
    Generate synthetic spatial interpolation scenarios with known ground truth.

    This class creates test scenarios for spatial interpolation methods by generating
    synthetic data with known mathematical functions. It supports both 2D and 3D cases,
    and can output data in griddata or non-griddata formats. The synthetic functions
    allow validation of interpolation methods against ground truth values.

    Parameters
    ----------
    n_dims : int, optional
        Dimensionality of the scenario (2 or 3). Default is 2.
    extent : list, optional
        Spatial extent of the domain.
        - For 2D: [x_min, x_max, y_min, y_max]
        - For 3D: [x_min, x_max, y_min, y_max, z_min, z_max]
        Default is [0, 1, 0, 1].
    griddata : bool, optional
        If True, generate regular grid format compatible with esi_griddata().
        If False, generate scattered points format for esi_nongriddata().
        Default is False.
    n_grid_points : int or tuple, optional
        Number of grid points per dimension. Can be:
        - int: Same number of points in all dimensions
        - tuple: Specific points per dimension (length must match n_dims)
        - None: Automatically set to (extent_max - extent_min + 1) per dimension
        Default is None.

    Attributes
    ----------
    n_dims : int
        Number of spatial dimensions (2 or 3).
    extent : list
        Spatial extent of the domain.
    griddata : bool
        Format flag for output data structure.
    n_grid_points : tuple
        Number of grid points per dimension.

    Examples
    --------
    Create a 2D scenario with default settings:

    >>> scenario = SyntheticScenario(n_dims=2, extent=[0, 100, 0, 150], griddata=True)
    >>> points, values, xi, reference = scenario.simulate_scenario(n_samples=300, seed=42)
    >>> scenario.plot_2d_scenario(points, xi, reference, theme='publication')

    Create a 3D scenario with custom grid resolution:

    >>> scenario = SyntheticScenario(n_dims=3, extent=[0, 10, 0, 10, 0, 5],
    ...                              griddata=False, n_grid_points=(50, 50, 25))
    >>> points, values, xi, reference = scenario.simulate_scenario(n_samples=500)

    See Also
    --------
    spatialize.gs.esi.esi_griddata : ESI for regular grids
    spatialize.gs.esi.esi_nongriddata : ESI for scattered points
    """
    def __init__(self,
                 n_dims=2,
                 extent=[0, 1, 0, 1],
                 griddata=False,
                 n_grid_points=None):
        # Assertions
        assert len(extent)==2*n_dims, f"{2*n_dims} values expected for extent."
        assert n_dims in [2, 3], f"{n_dims} dimensions not supported. 2D and 3D available."

        self.n_dims = n_dims
        self.extent = extent
        self.griddata = griddata

        # Number of grid points
        self.n_grid_points = n_grid_points or tuple([int(extent[i+1] - extent[i] + 1) 
                                                       for i in range(0, len(extent), 2)])
        # If single number, apply to all dimensions
        if isinstance(self.n_grid_points, int):
            self.n_grid_points = tuple([self.n_grid_points] * n_dims)

    def create_regular_grid(self):
        """
        Create a regular grid for interpolation.

        Generates a regular grid covering the spatial extent defined in the scenario.
        The output format depends on the `griddata` flag.

        Returns
        -------
        ndarray
            Grid points in the format specified by `self.griddata`:
            - If griddata=True (2D): shape (2, nx, ny) for use with esi_griddata()
            - If griddata=False (2D): shape (nx*ny, 2) array of [x, y] coordinates
            - If griddata=True (3D): shape (3, nx, ny, nz) for use with esi_griddata()
            - If griddata=False (3D): shape (nx*ny*nz, 3) array of [x, y, z] coordinates

        Examples
        --------
        >>> scenario = SyntheticScenario(n_dims=2, extent=[0, 10, 0, 20],
        ...                              griddata=True, n_grid_points=50)
        >>> xi = scenario.create_regular_grid()
        >>> xi.shape
        (2, 50, 50)
        """
        if self.n_dims == 2:
            x_min, x_max, y_min, y_max = self.extent
            nx, ny = self.n_grid_points
            
            if self.griddata:
                return np.mgrid[x_min:x_max:nx*1j, y_min:y_max:ny*1j]
            
            x_coords = np.linspace(x_min, x_max, nx)
            y_coords = np.linspace(y_min, y_max, ny)
            return np.array([(x, y) for x in x_coords for y in y_coords])
        
        else:       # 3D
            x_min, x_max, y_min, y_max, z_min, z_max = self.extent
            nx, ny, nz = self.n_grid_points
            
            if self.griddata:
                return np.mgrid[x_min:x_max:nx*1j, 
                               y_min:y_max:ny*1j, 
                               z_min:z_max:nz*1j]
            
            x_coords = np.linspace(x_min, x_max, nx)
            y_coords = np.linspace(y_min, y_max, ny)
            z_coords = np.linspace(z_min, z_max, nz)
            return np.array([(x, y, z) for x in x_coords for y in y_coords for z in z_coords])
    
    # Sample data generation
    def _normalize_coords(self, *coords):
        """
        Normalize coordinates to [0, 1] range based on extent.

        Parameters
        ----------
        *coords : array_like
            Variable number of coordinate arrays to normalize.

        Returns
        -------
        list
            List of normalized coordinate arrays.
        """
        normalized = []
        for i, coord in enumerate(coords):
            min_val = self.extent[2*i]
            max_val = self.extent[2*i + 1]
            normalized.append((coord - min_val) / (max_val - min_val))
        return normalized
    
    def cubic_func_2d(self, x, y):
        """
        Generate 2D synthetic test function values.

        Computes a smooth, non-linear test function with multiple local extrema.
        The function is designed to test interpolation performance with complex
        spatial patterns. Coordinates are automatically normalized to [0, 1].

        Parameters
        ----------
        x : array_like
            X-coordinates (will be normalized to extent).
        y : array_like
            Y-coordinates (will be normalized to extent).

        Returns
        -------
        ndarray
            Function values at the given coordinates.

        Notes
        -----
        The function formula is:
        f(x,y) = x(1-x) * cos(4πx) * sin²(4πy²)

        where x and y are first normalized to [0,1].
        """
        x, y = self._normalize_coords(x, y)
        return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2
    
    def cubic_func_3d(self, x, y, z):
        """
        Generate 3D synthetic test function values.

        Computes a smooth, non-linear 3D test function with multiple local extrema.
        The function extends the 2D version by adding variation in the z-dimension.
        Coordinates are automatically normalized to [0, 1].

        Parameters
        ----------
        x : array_like
            X-coordinates (will be normalized to extent).
        y : array_like
            Y-coordinates (will be normalized to extent).
        z : array_like
            Z-coordinates (will be normalized to extent).

        Returns
        -------
        ndarray
            Function values at the given coordinates.

        Notes
        -----
        The function formula is:
        f(x,y,z) = x(1-x) * cos(4πx) * sin²(4πy²) * cos(4πz)

        where x, y, and z are first normalized to [0,1].
        """
        x, y, z = self._normalize_coords(x, y, z)
        return (x * (1 - x) * np.cos(4 * np.pi * x) * 
                np.sin(4 * np.pi * y ** 2) ** 2 * 
                np.cos(4 * np.pi * z))

    def simulate_scenario(self, kind='cubic', n_samples=100, seed=None,
                          custom_func=None, custom_func_params=None):
        """
        Generate a complete synthetic interpolation scenario.

        Creates random sample points with known values and a reference grid with
        ground truth values for validation. This provides all inputs needed to
        test spatial interpolation methods.

        Parameters
        ----------
        kind : str, optional
            Type of synthetic function to use. Options:
            - 'cubic': Uses cubic_func_2d() or cubic_func_3d()
            Default is 'cubic'.
        n_samples : int, optional
            Number of random sample points to generate. Default is 100.
        seed : int, optional
            Random seed for reproducibility. If None, results will vary.
            Default is None.
        custom_func : callable, optional
            Custom function to use instead of built-in functions.
            Should accept coordinates as separate arguments (x, y) or (x, y, z).
            Default is None.
        custom_func_params : dict, optional
            Additional keyword arguments to pass to custom_func.
            Default is None.

        Returns
        -------
        points : ndarray, shape (n_samples, n_dims)
            Random sample point coordinates.
        values : ndarray, shape (n_samples,)
            Function values at sample points.
        xi : ndarray
            Interpolation grid locations (format depends on self.griddata).
        reference_values : ndarray
            Ground truth values at grid locations for validation.

        Raises
        ------
        ValueError
            If kind is not 'cubic' and no custom_func is provided.

        Examples
        --------
        >>> scenario = SyntheticScenario(n_dims=2, extent=[0, 100, 0, 150], griddata=True)
        >>> points, values, xi, reference = scenario.simulate_scenario(n_samples=300, seed=42)
        >>> print(f"Samples: {points.shape}, Grid: {xi.shape}")
        Samples: (300, 2), Grid: (2, 101, 151)

        Using a custom function:

        >>> def linear_func(x, y, slope=1.0):
        ...     return slope * (x + y)
        >>> scenario = SyntheticScenario(n_dims=2)
        >>> points, values, xi, ref = scenario.simulate_scenario(
        ...     n_samples=50, custom_func=linear_func, custom_func_params={'slope': 2.0})
        """
        if seed is not None:
            np.random.seed(seed)

        # Grid to make estimates
        xi = self.create_regular_grid()

        # Random locations for the samples
        ranges = [self.extent[i+1] - self.extent[i] for i in range(0, len(self.extent), 2)]
        offsets = [self.extent[i] for i in range(0, len(self.extent), 2)]
        points = np.random.random((n_samples, self.n_dims)) * ranges + offsets

        # Select function and apply to points and xi to generate values and reference_values
        if kind == 'cubic':
            func = self.cubic_func_2d if self.n_dims == 2 else self.cubic_func_3d
            func_params = {}
        elif custom_func:
            func = custom_func
            func_params = custom_func_params or {}
        else:
            raise ValueError(f"kind '{kind}' not available. Use 'cubic' or provide a custom function.")
        
        values = func(*points.T, **func_params)

        if self.griddata:
            reference_values = func(*xi, **func_params)
        else:
            reference_values = func(*xi.T, **func_params)
        
        return points, values, xi, reference_values
    
    def plot_2d_scenario(self, points, xi, reference_values,
                         theme='alges', cmap=None,
                         point_size=1.5, point_color='white',
                         figsize=(5, 6), dpi=100, title='Reference values and data points'):
        """
        Visualize 2D scenario with reference values and sample points.

        Creates a plot showing the ground truth function values across the domain
        with sample point locations overlaid. Useful for visualizing synthetic
        scenarios before and after interpolation.

        Parameters
        ----------
        points : ndarray, shape (n_samples, 2)
            Sample point coordinates from simulate_scenario().
        xi : ndarray
            Grid locations from simulate_scenario() (format depends on griddata).
        reference_values : ndarray
            Ground truth values from simulate_scenario().
        theme : str, optional
            Visualization theme. Options: 'alges', 'minimal', 'publication', 'whitegrid'.
            Default is 'alges'.
        cmap : str or Colormap, optional
            Matplotlib colormap. If None, uses theme default. Default is None.
        point_size : float, optional
            Size of sample point markers. Default is 1.5.
        point_color : str, optional
            Color of sample point markers. Default is 'white'.
        figsize : tuple, optional
            Figure size as (width, height) in inches. Default is (5, 6).
        dpi : int, optional
            Figure resolution in dots per inch. Default is 100.
        title : str, optional
            Plot title. Default is 'Reference values and data points'.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        ax : matplotlib.axes.Axes
            The created axes.

        Raises
        ------
        AssertionError
            If the scenario is not 2D.

        Examples
        --------
        >>> scenario = SyntheticScenario(n_dims=2, extent=[0, 100, 0, 150], griddata=True)
        >>> points, values, xi, reference = scenario.simulate_scenario(n_samples=200)
        >>> fig, ax = scenario.plot_2d_scenario(points, xi, reference, theme='publication')
        >>> plt.show()
        """
        assert self.n_dims == 2, "Only 2D scenarios supported."


        with PlotStyle(theme=theme, cmap=cmap) as style:
            fig, ax = plt.subplots(1,1, figsize=figsize, dpi=dpi)
            if self.griddata:
                plot_colormap_data(reference_values, ax=ax, xi_locations=xi,
                                griddata=self.griddata, cmap = style.cmap, extent=self.extent)
            else:
                plot_nongriddata(reference_values, xi_locations=xi, ax=ax, cmap = style.cmap)
            ax.scatter(points[:, 0], points[:, 1], s=point_size, color=point_color)
            ax.set_title(title)
            return fig, ax


class PrecipitationCaseStudy:
    """
    Real-world precipitation case study with visualization utilities.

    This class provides a complete workflow for a 2.5D (x, y, elevation) precipitation
    interpolation case study. It loads precipitation measurements from weather stations,
    defines interpolation locations with elevation data, and provides formatted
    visualization methods for results.

    The case study covers three dates with varying precipitation patterns and is
    useful for demonstrating ESI methods on real-world environmental data.

    Attributes
    ----------
    locs : pd.DataFrame
        Interpolation locations with columns ['X', 'Y', 'Z'] (UTM coordinates + elevation).
    data : pd.DataFrame
        Precipitation sample data with columns ['x', 'y', 'z'] plus date columns.
    study_area : geopandas.GeoDataFrame
        Shapefile boundary of the study area for plotting.
    dates : list
        List of date strings for the case study (typically 3 dates).
    locs_array : xarray.Dataset
        Interpolation locations as xarray for plotting.
    extent : list
        Spatial extent as [xmin, xmax, ymin, ymax].
    data_cmap : matplotlib.colors.Colormap
        Colormap for precipitation values.
    locs_cmap : matplotlib.colors.Colormap
        Colormap for elevation values.
    precision_cmap : matplotlib.colors.Colormap
        Colormap for precision/uncertainty values.

    Methods
    -------
    model_inputs()
        Get formatted inputs for ESI functions (points, values, xi).
    plot_input_data()
        Visualize precipitation samples for all dates.
    plot_interpolation_locations()
        Visualize interpolation grid with elevation.
    plot_esi_results(results, parameters)
        Create comprehensive 3x3 plot of inputs, estimates, and precision.
    plot_model_comparison(results, date, interpolators, names)
        Compare multiple interpolation methods side-by-side.

    Examples
    --------
    Basic workflow for the precipitation case study:

    >>> case_study = PrecipitationCaseStudy()
    >>> points, values, xi = case_study.model_inputs()
    >>>
    >>> # Run ESI for each date
    >>> results = {}
    >>> for date in case_study.dates:
    ...     result = esi_nongriddata(points[date], values[date], xi,
    ...                              local_interpolator='idw', exponent=2.0)
    ...     results[date] = result
    >>>
    >>> # Visualize results
    >>> case_study.plot_input_data()
    >>> case_study.plot_esi_results(results_df, parameters_df)

    Notes
    -----
    This case study requires the following data files in ../data/:
    - PP_locations.csv: Interpolation grid with elevation
    - PP_samples.csv: Precipitation measurements at stations
    - PP_Basin/Basin_UTM.shp: Study area boundary shapefile

    The data uses UTM coordinates (meters) and elevation in meters above sea level.
    Precipitation values are in millimeters (mm).

    See Also
    --------
    SyntheticScenario : Generate synthetic test scenarios
    spatialize.gs.esi.esi_nongriddata : ESI for scattered points
    """
    def __init__(self):
        self._load_data()
        self._setup_environment()

    def _load_data(self):
        """
        Load precipitation case study data files.

        Loads interpolation locations, precipitation samples, and study area
        boundary from the data directory.
        """
        import geopandas as gpd

        script_dir = Path(__file__).parent

        # Interpolation locations for the 'xi' input:
        self.locs = pd.read_csv(script_dir / '../data/PP_locations.csv')

        # Precipitation data:
        self.data = pd.read_csv(script_dir / '../data/PP_samples.csv')

        # Outline of study area
        self.study_area = gpd.read_file(script_dir / '../data/PP_Basin/Basin_UTM.shp')

    def _setup_environment(self,
                           locs_cmap=None,
                           data_cmap=None,
                           precision_cmap=None):
        """
        Set up plot configurations and colormaps.

        Configures matplotlib settings for consistent visualization and creates
        custom colormaps for precipitation, elevation, and precision plots.

        Parameters
        ----------
        locs_cmap : matplotlib.colors.Colormap, optional
            Custom colormap for elevation. If None, uses seaborn blend. Default is None.
        data_cmap : matplotlib.colors.Colormap, optional
            Custom colormap for precipitation. If None, uses seaborn blend. Default is None.
        precision_cmap : matplotlib.colors.Colormap, optional
            Custom colormap for precision. If None, uses seaborn blend. Default is None.
        """
        # Matplotlib configuration
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': 'DejaVu Serif',
            'grid.alpha': 0.6,
            'axes.titleweight': 'demibold',
            'axes.titlesize': 11,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'figure.titleweight': 'bold',
            'figure.titlesize': 13,
            'savefig.bbox': 'tight'
            })
        
        # List of studied dates (to iterate on)
        self.dates = list(self.data.columns)[3:6]

        # For imshow plots
        self.locs_array = self.locs.set_index(['X', 'Y']).to_xarray()
        self.extent = calculate_extent(self.locs)

        # Colormaps
        import seaborn as sns

        self.data_cmap = data_cmap or sns.color_palette("blend:#ffe370,#84ffb8,#38b2ff,#009abe,#006a83,#004252", as_cmap=True)
        self.data_cmap.set_under('#ffce06')
        self.data_cmap.set_over('#00242d')

        self.locs_cmap = locs_cmap or sns.color_palette("blend:#80a74f,#bddb97,#e3cfb2,#d6b78c,#c9a066,#ba8842,#946c35,#825f2e,#332512", as_cmap=True)

        self.precision_cmap = precision_cmap or sns.color_palette("blend:#4b7f52,#7dd181,#b6f9c9,#ffdf80,#f8a07f,#f57d4e,#d2430b", as_cmap=True)
        self.precision_cmap.set_over('#b93007')
    
    def model_inputs(self):
        """
        Get formatted inputs for ESI interpolation functions.

        Extracts and formats the data into the structure expected by ESI functions.
        Sample points and values are organized by date, while interpolation locations
        are shared across all dates.

        Returns
        -------
        points : dict
            Dictionary mapping date strings to sample coordinate arrays.
            Each array has shape (n_samples, 3) with columns [x, y, z].
        values : dict
            Dictionary mapping date strings to precipitation value arrays.
            Each array has shape (n_samples,) with precipitation in mm.
        xi : ndarray, shape (n_locations, 3)
            Interpolation locations with columns [X, Y, Z] (UTM + elevation).

        Examples
        --------
        >>> case_study = PrecipitationCaseStudy()
        >>> points, values, xi = case_study.model_inputs()
        >>> print(f"Dates: {list(points.keys())}")
        >>> print(f"Samples for first date: {points[case_study.dates[0]].shape}")
        >>> print(f"Interpolation locations: {xi.shape}")
        """
        # Interpolation locations:
        xi = self.locs[['X', 'Y', 'Z']].values

        # Samples:
        points = {}
        values = {}

        for date in self.dates:
            mask = ~np.isnan(self.data[date])        # Mask for non-NaN values
            points[date] = self.data[['x', 'y', 'z']][mask].values
            values[date] = self.data[date][mask].values

        return points, values, xi
    
    def _plot_format(self, ax):
        """
        Apply consistent formatting to precipitation case study plots.

        Sets axis limits, tick locations, and overlays the study area boundary.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to format.
        """
        ax.set_xticks(ticks=[250000, 310000, 370000, 430000], labels=['250000', '310000', '370000', '430000'])
        ax.set_yticks(ticks=[6200000, 6240000, 6280000, 6320000, 6360000], labels=['6200000', '6240000', '6280000', '6320000', '6360000'])
        ax.set_xlim([250000 - 6000, 430000 + 6000])
        ax.set_ylim([6200000, 6360000])

        self.study_area.boundary.plot(ax=ax, color='black', linewidth=0.5)

    def plot_input_data(self):
        """
        Plot precipitation sample data for all dates.

        Creates a figure with three subplots showing the spatial distribution of
        precipitation measurements for each date in the case study.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        axs : array of matplotlib.axes.Axes
            Array of axes for each date subplot.

        Examples
        --------
        >>> case_study = PrecipitationCaseStudy()
        >>> fig, axs = case_study.plot_input_data()
        >>> plt.show()
        """
        fig, axs = plt.subplots(1, 3, figsize=(9.8, 3), sharey=True, layout='compressed')

        for i, date in enumerate(self.dates):
            date = self.dates[i]
            scatter = axs[i].scatter(self.data['x'], self.data['y'],
                                    c=self.data[date], cmap=self.data_cmap,
                                    clim=(0, 100), zorder=3)
            self._plot_format(axs[i])
            axs[i].set_title(date)

        fig.suptitle('Precipitation Data')
        axs[0].set_ylabel('UTM North')
        axs[1].set_xlabel('UTM East')

        cbar1 = plt.colorbar(scatter, ax=axs[2], aspect=10)
        cbar1.set_label('Precipitation [mm]', fontsize=11)

        return fig, axs
    
    def plot_interpolation_locations(self):
        """
        Plot interpolation grid with elevation values.

        Visualizes the spatial distribution of interpolation locations colored by
        elevation, providing context for the topography of the study area.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        ax : matplotlib.axes.Axes
            The created axes.

        Examples
        --------
        >>> case_study = PrecipitationCaseStudy()
        >>> fig, ax = case_study.plot_interpolation_locations()
        >>> plt.show()
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), layout='compressed')

        image = ax.imshow(np.flipud(self.locs_array.Z.T), cmap=self.locs_cmap,
                          clim=(0, 6000), zorder=1, extent=self.extent)
        self._plot_format(ax)

        fig.suptitle('Interpolation Locations')
        ax.set_ylabel('UTM North')
        ax.set_xlabel('UTM East')

        cbar = fig.colorbar(image, aspect=10)
        cbar.set_label('Elevation [m]', labelpad=8, fontsize=11)

        return fig, ax
    
    def plot_esi_results(self, results,
                         parameters=None,
                         local_interpolator='idw',
                         precision_function='Operational Error',
                         fig_title='ESI Results',
                         dpi=100):
        """
        Generate comprehensive 3x3 visualization of ESI results.

        Creates a publication-quality figure showing input data, ESI estimates, and
        precision maps for all three dates in the case study. Each row represents
        a different aspect (inputs, estimates, precision) and each column represents
        a different date.

        Parameters
        ----------
        results : pd.DataFrame
            DataFrame with MultiIndex columns (date, 'value'/'precision') and
            index columns ['X', 'Y'] matching interpolation locations.
        parameters : pd.DataFrame
            DataFrame with dates as index and parameter columns (e.g., 'alpha',
            'exponent' for IDW or 'model', 'nugget', 'range' for Kriging).
        local_interpolator : str, optional
            Type of local interpolator used ('idw' or 'kriging'). Affects parameter
            annotation formatting. Default is 'idw'.
        precision_function : str, optional
            Name of precision metric for colorbar label. Default is 'Operational Error'.
        fig_title : str, optional
            Overall figure title. Default is 'ESI Results'.
        dpi : int, optional
            Figure resolution in dots per inch. Default is 100.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        axs : ndarray of matplotlib.axes.Axes
            3x3 array of axes.

        Examples
        --------
        >>> case_study = PrecipitationCaseStudy()
        >>> # ... run ESI for all dates and organize results ...
        >>> fig, axs = case_study.plot_esi_results(results_df, params_df)
        >>> plt.savefig('esi_results.png', dpi=300, bbox_inches='tight')
        """
        # Transform data for plots
        results_array = results.set_index(['X', 'Y']).to_xarray()

        fig, axs = plt.subplots(3, 3, figsize = (11, 8.4), dpi = dpi, sharex = True, sharey = True, layout = 'compressed')
        fig.suptitle(fig_title, fontsize = 14)

        for j, date in enumerate(self.dates):
            # Format
            axs[0,j].set_title(date, fontsize = 12)
            for i in range(0,3):
                self._plot_format(axs[i,j])
            
            # Input data and interpolation surface plot
            image = axs[0,j].imshow(np.flipud(self.locs_array.Z.T),
                                    cmap = self.locs_cmap, clim = (0, 6000),
                                    zorder = 1, extent = self.extent)
            axs[0,j].scatter(self.data['x'], self.data['y'],
                             c = self.data[date], cmap = self.data_cmap, clim = (0, 100),
                             edgecolors = 'white', linewidth = 0.5, zorder = 3)
            
            # ESI Result plot
            result_plot = axs[1,j].imshow(np.flipud(results_array[(date,'value')].T),
                                        cmap = self.data_cmap, clim = (0, 100), zorder = 1, extent = self.extent)
            
            # Annotate parameters
            if parameters is not None:
                if local_interpolator=='idw':
                    axs[1,j].text(252000, 6340000,
                                f"alpha = {parameters.loc[date, 'alpha']}\nexp = {parameters.loc[date, 'exponent']}",
                                fontsize = 8.5)
                elif local_interpolator == 'kriging':
                    axs[1,j].text(252000, 6205000,
                                f"alpha = {parameters.loc[date, 'alpha']}\nmodel = {parameters.loc[date, 'model']}\nnugget = {parameters.loc[date, 'nugget']}\nrange = {parameters.loc[date, 'range']}",
                                fontsize = 7.5)

            # Precision plot
            precision_plot = axs[2,j].imshow(np.flipud(results_array[(date, 'precision')].T),
                                        cmap = self.precision_cmap, clim = (0, 0.3),
                                        zorder = 1, extent = self.extent)

        # Titles and labels 
        axs[0,0].set_ylabel('Input Data', labelpad = 25, fontweight = 'bold')
        axs[1,0].set_ylabel('ESI Estimates', labelpad = 25, fontweight = 'bold')
        axs[2,0].set_ylabel('ESI Precision', labelpad = 25, fontweight = 'bold')

        axs[2,1].set_xlabel('UTM East', labelpad = 8, fontsize = 11)
        fig.text(0.035, 0.5, 'UTM North', rotation = 'vertical', fontsize = 11, ha = 'center', va = 'center')

        # Colorbars
        cbar = plt.colorbar(image, ax = axs[0,2], aspect = 10)
        cbar.set_label('Elevation [m]', labelpad = 8, fontsize = 11)

        cbar1 = plt.colorbar(result_plot, ax = axs[1,2], aspect = 10)
        cbar1.set_label('Precipitation [mm]', labelpad = 8, fontsize = 11)

        cbar2 = plt.colorbar(precision_plot, ax = axs[2,2], aspect = 10)
        cbar2.set_label(precision_function, labelpad = 8, fontsize = 11)

        return fig, axs
    
    def plot_model_comparison(self, results, date,
                              interpolators=['nearest', 'rbf', 'kriging', 'esi'],
                              names=['Nearest Neighbor', 'RBF', 'Kriging', 'ESI'],
                              dpi=100):
        """
        Compare multiple interpolation methods side-by-side for a single date.

        Creates a visualization comparing different interpolation approaches, showing
        the input data and elevation on the left and interpolation results from
        different methods on the right.

        Parameters
        ----------
        results : pd.DataFrame
            DataFrame with MultiIndex columns (interpolator, date, 'value') and
            index columns ['X', 'Y'].
        date : str
            Date to visualize (must be one of self.dates).
        interpolators : list of str, optional
            List of interpolator names matching first level of results columns.
            Default is ['nearest', 'rbf', 'kriging', 'esi'].
        names : list of str, optional
            Display names for each interpolator (same order as interpolators).
            Default is ['Nearest Neighbor', 'RBF', 'Kriging', 'ESI'].
        dpi : int, optional
            Figure resolution in dots per inch. Default is 100.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        subfigs : array of matplotlib.figure.SubFigure
            Array containing [input_subfig, results_subfig].
        ax0 : matplotlib.axes.Axes
            Axes for input data visualization.
        axs : ndarray of matplotlib.axes.Axes
            2xN array of axes for interpolation method results.

        Examples
        --------
        >>> case_study = PrecipitationCaseStudy()
        >>> # ... run multiple interpolation methods and organize results ...
        >>> fig, subfigs, ax0, axs = case_study.plot_model_comparison(
        ...     results_df, date='2021-01-15',
        ...     interpolators=['idw', 'kriging', 'esi'],
        ...     names=['IDW', 'Kriging', 'ESI'])
        >>> plt.show()
        """
        results_array = results.set_index(['X', 'Y']).to_xarray()
        n_models = len(interpolators)

        #for date in self.dates:
        fig = plt.figure(layout = 'constrained', figsize = (8+n_models, 5.8), dpi=dpi)
        fig.suptitle(f'Estimates for {date}', fontsize = 16)
        subfigs = fig.subfigures(1, 2, width_ratios = [1, (2.2/3)*(n_models/2)], wspace = 0.03)

        # === PRECIPITATION DATA + INTERPOLATION LOCATIONS ===
        ax0 = subfigs[0].add_subplot(111)
        self._plot_format(ax0)
        ax0.text(252000, 6340000, f"min = {self.data[date].min()}\nmax = {self.data[date].max()}", fontsize = 9, zorder = 4)
        ax0.set_title('Inputs', fontsize = 13, pad = 8)

        # Interpolation locations:
        image = ax0.imshow(np.flipud(self.locs_array.Z.T),
                        cmap = self.locs_cmap, clim = (0, 6000),
                        zorder = 1, extent = self.extent)
        
        # Precipitation data:
        scatter = ax0.scatter(self.data['x'], self.data['y'],
                            c = self.data[date], cmap = self.data_cmap, clim = (0, 100),
                            edgecolors = 'white', linewidth = 0.6, zorder = 3)

        # === INTERPOLATORS ===
        axs = subfigs[1].subplots(nrows = 2, ncols = int(n_models / 2), sharex = 'all', sharey = 'all')

        # Plots
        for i, ax in enumerate(axs.flatten()):
            model_results = results_array[(interpolators[i], date, 'value')]
            ax.imshow(np.flipud(model_results.T),
                        cmap = self.data_cmap, clim = (0, 100),
                        zorder = 1, extent = self.extent)
            ax.set_title(names[i], fontsize = 13, pad = 8)
            self._plot_format(ax)
            ax.text(252000, 6340000, f"min = {float(model_results.min()):.1f}\nmax = {float(model_results.max()):.1f}", fontsize = 8)

        # Colorbars
        cbar = plt.colorbar(scatter, ax = ax0, aspect = 14, location = 'bottom', extend = 'both')
        cbar.set_label('Precipitation [mm]')
        cbar1 = plt.colorbar(image, ax = ax0, aspect = 14, location = 'bottom', pad = 0.12)
        cbar1.set_label('Elevation [m]')

        # Labels
        ax0.set_xlabel('UTM East', fontsize = 11)
        ax0.set_ylabel('UTM North', fontsize = 11)
        axs[1,1].set_xlabel('UTM East', fontsize = 11)
        subfigs[1].supylabel('UTM North', fontsize = 11)

        return fig, subfigs, ax0, axs

class SuppressOutput:
    """
    Context manager to temporarily suppress stdout output.

    Redirects stdout to /dev/null for the duration of the context, suppressing
    all print statements and stdout output. Useful for silencing verbose library
    functions during batch processing.

    Examples
    --------
    >>> with SuppressOutput():
    ...     print("This will not be displayed")
    ...     # Any function calls here will have stdout suppressed
    >>> print("This will be displayed normally")
    This will be displayed normally

    Warnings
    --------
    This suppresses ALL stdout within the context, including error messages.
    Use with caution and ensure proper error handling.
    """
    def __enter__(self):
        """Enter the context and redirect stdout to /dev/null."""
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        return self

    def __exit__(self, *args):
        """Exit the context and restore stdout."""
        sys.stdout.close()
        sys.stdout = self._original_stdout
