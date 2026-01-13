# yabplot: yet another brain plot

<p align="center">
  <img src="assets/yabplot_logo.png" alt="logo width="200"/>
</p>

**yabplot** is a Python library for creating beautiful, publication-quality 3D brain visualizations. it supports plotting cortical regions, subcortical structures, and white matter bundles, built on top of [PyVista](https://docs.pyvista.org/).

the idea is simple. while there are already amazing visualization tools available, they often focus on specific domains—using one tool for white matter tracts and another for cortical surfaces inevitably leads to inconsistent styles. i wanted a unified, simple-to-use tool that enables me (and hopefully others) to perform most brain visualizations in a single place. recognizing that neuroscience evolves daily, i designed **yabplot** to be modular: it supports standard pre-packaged atlases out of the box, but easily accepts any custom parcellation or tractography dataset you might need.

## features

* **pre-existing atlases:** access many commonly used atlases (schaefer, brainnetome, hcp) on demand.
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

please refer to the documentation and examples for comprehensive guides.

```python
import yabplot as yab
import numpy as np

# plot random data on cortical regions
regions = yab.get_atlas_regions('schaefer_400', 'cortical')
data = np.random.rand(len(regions))
yab.plot_cortical(data=data, atlas='schaefer_400', cmap='viridis')

# plot values for specific subcortical regions
data = {'Left_Amygdala': 0.8, 'Right_Thalamus': 0.5}
yab.plot_subcortical(data=data, atlas='aseg', views=['left_lateral', 'superior'])

# plot values for specific white matter bundles
data = {'FMaj': 0.2, 'FMin': -0.3}
yab.plot_tracts(data=data, atlas='xtract_tiny', style='matte')

```

## acknowledgements

yabplot relies on the extensive work of the neuroimaging community. if you use these atlases in your work, please cite the original authors.