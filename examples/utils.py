import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import sys
import os
from pathlib import Path

import import_helper
from spatialize.viz import plot_colormap_data, PlotStyle

class SyntheticScenario:
    def __init__(self,
                 n_dims=2,
                 extent=[0, 1, 0, 1], 
                 griddata=False, 
                 n_grid_points=None):
        """
        n_dims: dimensionality (2 or 3).
        extent: [x_min, x_max, y_min, y_max] if n_dims = 2
                [x_min, x_max, y_min, y_max, z_min, z_max] if n_dims = 3
        n_grid_points: int or tuple. Puntos por dimensión. Si None, usa (extent_max - extent_min + 1)
        """
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
        """Creates regular grid in griddata or non-griddata formats."""
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
        """Normaliza coordenadas a [0, 1]"""
        normalized = []
        for i, coord in enumerate(coords):
            min_val = self.extent[2*i]
            max_val = self.extent[2*i + 1]
            normalized.append((coord - min_val) / (max_val - min_val))
        return normalized
    
    def cubic_func_2d(self, x, y):
        """A kind of 'cubic' function."""
        x, y = self._normalize_coords(x, y)
        return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2
    
    def cubic_func_3d(self, x, y, z):
        """A kind of 'cubic' function in 3D."""
        x, y, z = self._normalize_coords(x, y, z)
        return (x * (1 - x) * np.cos(4 * np.pi * x) * 
                np.sin(4 * np.pi * y ** 2) ** 2 * 
                np.cos(4 * np.pi * z))

    # Simulate scenario
    def simulate_scenario(self, kind='cubic', n_samples=100, seed=None,
                          custom_func=None, custom_func_params=None):
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
        Visualize 2D scenario with reference values and samples.

        points, xi, reference_values: from output of simulate_scenario()
        """
        assert self.n_dims == 2, "Only 2D scenarios suported."


        with PlotStyle(theme=theme, cmap=cmap) as style:
            fig, ax = plt.subplots(1,1, figsize=figsize, dpi=dpi)
            plot_colormap_data(reference_values, ax=ax, xi_locations=xi,
                               griddata=self.griddata, cmap = style.cmap, extent=self.extent,)
            ax.scatter(points[:, 0], points[:, 1], s=point_size, color=point_color)
            ax.set_title(title)
            return fig, ax

    
class PrecipitationCaseStudy:
    def __init__(self):

        self._load_data()
        self._setup_environment()


    def _load_data(self):
        import geopandas as gpd
        #self.gpd = gpd

        script_dir = Path(__file__).parent
        
        # Interpolation locations for the 'xi' input:
        #locs = pd.read_csv('../data/PP_locations.csv', sep = ',')
        self.locs = pd.read_csv(script_dir / '../data/PP_locations.csv')

        # Precipitation data:
        #data = pd.read_csv('../data/PP_samples.csv', sep = ",", header = 0)
        self.data = pd.read_csv(script_dir / '../data/PP_samples.csv')

        # Outline of study area
        self.study_area = gpd.read_file(script_dir / '../data/PP_Basin/Basin_UTM.shp')        # study area shapefile

    def _setup_environment(self,
                           locs_cmap = None,
                           data_cmap = None,
                           precision_cmap = None):
        """Sets up plot configurations, environment variables and colormaps."""
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
        self.extent = [self.locs.X.min(), self.locs.X.max(), self.locs.Y.min(), self.locs.Y.max()]

        # Colormaps
        import seaborn as sns

        self.data_cmap = data_cmap or sns.color_palette("blend:#ffe370,#84ffb8,#38b2ff,#009abe,#006a83,#004252", as_cmap=True)
        self.data_cmap.set_under('#ffce06')
        self.data_cmap.set_over('#00242d')

        self.locs_cmap = locs_cmap or sns.color_palette("blend:#80a74f,#bddb97,#e3cfb2,#d6b78c,#c9a066,#ba8842,#946c35,#825f2e,#332512", as_cmap=True)

        self.precision_cmap = precision_cmap or sns.color_palette("blend:#4b7f52,#7dd181,#b6f9c9,#ffdf80,#f8a07f,#f57d4e,#d2430b", as_cmap=True)
        self.precision_cmap.set_over('#b93007')
    
    def model_inputs(self):
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
        """ Sets plot limits, ticks, and plots study area """
        ax.set_xticks(ticks=[250000, 310000, 370000, 430000], labels=['250000', '310000', '370000', '430000'])
        ax.set_yticks(ticks=[6200000, 6240000, 6280000, 6320000, 6360000], labels=['6200000', '6240000', '6280000', '6320000', '6360000'])
        ax.set_xlim([250000 - 6000, 430000 + 6000])
        ax.set_ylim([6200000, 6360000])

        self.study_area.boundary.plot(ax=ax, color='black', linewidth=0.5)

    def plot_input_data(self):
        """Plots sample data for all three dates."""
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
        """Plots interpolation locations, representing """
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
    
    def plot_esi_results(self, results, parameters,
                         local_interpolator='idw',
                         precision_function = 'Operational Error',
                         fig_title = 'ESI Results',
                         dpi = 100):
        """Generates a 3x3 plot showing reference values, esi estimates and esi precision for all three dates."""
        # Setup colormaps
        """locs_cmap = locs_cmap or self.locs_cmap
        data_cmap = data_cmap or self.data_cmap
        precision_cmap = precision_cmap or self.precision_cmap"""

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
                              interpolators = ['nearest', 'rbf', 'kriging', 'esi'],
                              names = ['Nearest Neighbor',  'RBF', 'Kriging', 'ESI'],
                              dpi=100):
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

# Redirige stdout temporalmente
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        return self
    
    def __exit__(self, *args):
        sys.stdout.close()
        sys.stdout = self._original_stdout
