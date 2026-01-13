# yabplot: yet another brain plot

![logo](assets/yabplot_logo.png)

[![PyPI version](https://img.shields.io/pypi/v/yabplot.svg)](https://pypi.org/project/yabplot/)
[![Docs](https://github.com/teanijarv/yabplot/actions/workflows/docs.yml/badge.svg)](https://teanijarv.github.io/yabplot/)
[![Tests](https://github.com/teanijarv/yabplot/actions/workflows/tests.yml/badge.svg)](https://github.com/teanijarv/yabplot/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18237144.svg)](https://doi.org/10.5281/zenodo.18237144)

**yabplot** is a Python library for creating beautiful, publication-quality 3D brain visualizations. it supports plotting cortical regions, subcortical structures, and white matter bundles.

the idea is simple. while there are already amazing visualization tools available, they often focus on specific domains—using one tool for white matter tracts and another for cortical surfaces inevitably leads to inconsistent styles. i wanted a unified, simple-to-use tool that enables me (and hopefully others) to perform most brain visualizations in a single place. recognizing that neuroscience evolves daily, i designed **yabplot** to be modular: it supports standard pre-packaged atlases out of the box, but easily accepts any custom parcellation or tractography dataset you might need.

## features

* **pre-existing atlases:** access many commonly used atlases (schaefer2018, brainnetome, aparc, aseg, musus100, xtract, etc) on demand.
* **simple to use:** plug-n-play functions for cortex, subcortex, and tracts with a unified API.
* **custom atlases:** easily use your own parcellations, segmentations (.nii/.gii), or tractograms (.trk).
* **flexible inputs:** accepts data as dictionaries (for partial mapping) or arrays (for strict mapping).

## installation

```bash
uv add yabplot
```
or
```bash
pip install yabplot
```

dependencies: python 3.11 with ipywidgets, nibabel, pandas, pooch, pyvista, scikit-image, trame, trame-vtk, trame-vuetify

## quick start

please refer to the [documentation](https://teanijarv.github.io/yabplot/) for more comprehensive guides.

```python
import yabplot as yab
import numpy as np

# see available cortical atlases
atlases = yab.get_available_resources(category='cortical')

# see the region names within the aseg atlas
regions = yab.get_atlas_regions(atlas='aseg', category='subcortical')

# plot data on cortical regions
data = np.arange(0, 1, 0.001)
yab.plot_cortical(data=data, atlas='schaefer_1000', figsize=(600, 300),
                  cmap='viridis', vminmax=[0, 1], style='default',
                  views=['left_lateral', 'superior', 'right_lateral'])


# plot values for specific subcortical regions
data = {'Left_Amygdala': 0.8, 'Right_Hippocampus': 0.5, 
        'Right_Thalamus': -0.5, 'Left_Putamen': -1}
yab.plot_subcortical(data=data, atlas='aseg', figsize=(600, 450), layout=(2, 2),
                     views=['superior', 'anterior', 'left_lateral', 'right_lateral'], 
                     cmap='coolwarm', vminmax=[-1, 1], style='matte')

# plot data on white matter bundles
regions = yab.get_atlas_regions(atlas='xtract_tiny', category='tracts')
data = np.arange(0, len(regions))
yab.plot_tracts(data=data, atlas='xtract_tiny', figsize=(600, 300), 
                views=['superior', 'anterior', 'left_lateral'], nan_color='#cccccc', 
                bmesh_type='fsaverage', style='default', cmap='plasma')

```

![examples](assets/examples.png)

## acknowledgements

yabplot relies on the extensive work of the neuroimaging community. if you use these atlases in your work, please cite the original authors. if you use this package for any scientific work, please cite the DOI (see more info on [Zenodo](https://doi.org/10.5281/zenodo.18237144)).