# Spatialize Examples

Worked examples and tutorials for the Spatialize library, grouped by topic. Each page is a runnable Jupyter notebook.

## Introductory

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} ESI Overview
:link: examples/introductory/esi_overview
:link-type: doc
:img-top: _static/thumbnails/esi_overview.png

A conceptual introduction to Ensemble Spatial Interpolation.
:::

:::{grid-item-card} ESS Overview
:link: examples/introductory/ess_overview
:link-type: doc
:img-top: _static/thumbnails/ess_overview.png

A first look at Ensemble Spatial Simulation.
:::

:::{grid-item-card} SPA Overview
:link: examples/introductory/spa_overview
:link-type: doc
:img-top: _static/thumbnails/spa_overview.png

Introducing the spatial analysis utilities module.
:::

::::

## ESI Fundamentals

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Gridded Interpolation
:link: examples/esi_fundamentals/esi_griddata
:link-type: doc
:img-top: _static/thumbnails/esi_griddata.png

Interpolating scattered data onto a regular grid with `esi_griddata`.
:::

:::{grid-item-card} Non-gridded Interpolation
:link: examples/esi_fundamentals/esi_nongriddata
:link-type: doc
:img-top: _static/thumbnails/esi_nongriddata.png

Interpolating at arbitrary scattered target locations with `esi_nongriddata`.
:::

:::{grid-item-card} Hyperparameter Search
:link: examples/esi_fundamentals/esi_hparams_search
:link-type: doc
:img-top: _static/thumbnails/esi_hparams_search.png

Tuning ESI parameters with k-fold cross-validated grid search.
:::

:::{grid-item-card} Pareto Optimization
:link: examples/esi_fundamentals/esi_pareto_optimization
:link-type: doc
:img-top: _static/thumbnails/esi_pareto_optimization.png

Jointly optimizing CV error against an empirical robustness bound.
:::

:::{grid-item-card} Precision Metrics
:link: examples/esi_fundamentals/esi_precision
:link-type: doc
:img-top: _static/thumbnails/esi_precision.png

Understanding and customizing ESI precision/error metrics.
:::

::::

## ESI Implementations

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 2.5D Interpolation
:link: examples/esi_implementations/esi_2.5d
:link-type: doc
:img-top: _static/thumbnails/esi_2.5d.png

Interpolating 2D spatial data with an elevation dimension.
:::

:::{grid-item-card} Adaptive ESI (2D)
:link: examples/esi_implementations/adaptive_esi_2d
:link-type: doc
:img-top: _static/thumbnails/adaptive_esi_2d.png

Cell-by-cell parameter optimization for anisotropic 2D fields.
:::

:::{grid-item-card} Adaptive ESI (2.5D)
:link: examples/esi_implementations/adaptive_esi_2.5d
:link-type: doc
:img-top: _static/thumbnails/adaptive_esi_2.5d.png

Adaptive ESI extended to 2D-plus-elevation data.
:::

:::{grid-item-card} Categorical ESI
:link: examples/esi_implementations/categorical_esi
:link-type: doc
:img-top: _static/thumbnails/categorical_esi.png

Ensemble spatial interpolation for categorical/classification data.
:::

::::

## How-To

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Custom Precision Functions
:link: examples/how_to/custom_esi_precision
:link-type: doc
:img-top: _static/thumbnails/custom_esi_precision.png

Defining and plugging in custom precision functions beyond the defaults.
:::

::::

## Utilities

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Empirical Tools
:link: examples/utilities/empirical_tools
:link-type: doc
:img-top: _static/thumbnails/empirical_tools.png

Fitting and querying empirical probabilistic models over ensemble members.
:::

:::{grid-item-card} Evaluation Tools
:link: examples/utilities/evaluation_tools
:link-type: doc
:img-top: _static/thumbnails/evaluation_tools.png

Benchmarking helpers: synthetic scenarios, metrics, and baselines.
:::

:::{grid-item-card} Visualization Tools
:link: examples/utilities/visualization_tools
:link-type: doc
:img-top: _static/thumbnails/visualization_tools.png

Themed plotting utilities for spatial data and results.
:::

::::

```{toctree}
:maxdepth: 1
:hidden:

examples/introductory/esi_overview
examples/introductory/ess_overview
examples/introductory/spa_overview
examples/esi_fundamentals/esi_griddata
examples/esi_fundamentals/esi_nongriddata
examples/esi_fundamentals/esi_hparams_search
examples/esi_fundamentals/esi_pareto_optimization
examples/esi_fundamentals/esi_precision
examples/esi_implementations/esi_2.5d
examples/esi_implementations/adaptive_esi_2d
examples/esi_implementations/adaptive_esi_2.5d
examples/esi_implementations/categorical_esi
examples/how_to/custom_esi_precision
examples/utilities/empirical_tools
examples/utilities/evaluation_tools
examples/utilities/visualization_tools
```
