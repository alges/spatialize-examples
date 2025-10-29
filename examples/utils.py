import numpy as np
import matplotlib.pyplot as plt

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

    
