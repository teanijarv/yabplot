---
title: 'yabplot: yet another brain plot for unified neuroimaging visualization in Python'
tags:
  - Python
  - neuroimaging
  - visualization
  - neuroscience
  - 3D
authors:
  - name: Toomas Erik Anijärv
    orcid: 0000-0002-3650-4230
    affiliation: 1
affiliations:
 - name: Clinical Memory Research Unit, Department of Clinical Sciences Malmö, Faculty of Medicine, Lund University, Lund, Sweden
   index: 1
date: 14 May 2026
bibliography: paper.bib
---

# Summary

Neuroimaging analyses often produce results in several anatomical representations. A single study may include values defined in atlas-based regions, such as cortical parcels, subcortical structures, or white-matter bundles, as well as continuous voxel-wise and vertex-wise maps and region-to-region connectivity matrices. Communicating these results requires figures that are anatomically interpretable, visually consistent, and reproducible from the same computational workflow used to generate the data.

`yabplot` (yet another brain plot) is an open-source Python package for creating three-dimensional neuroimaging visualizations across these representations. It provides a unified interface for plotting cortical parcellations, vertex-wise cortical data, voxel-wise volumetric data, subcortical structures, white-matter bundles, tractometry results, and connectome graphs. The package is intended for researchers who want to create publication-ready figures directly from Python scripts or Jupyter notebooks, without switching between multiple specialized graphical applications. Supported inputs depend on the plotting function and include NIfTI images, surface files, tractography files, arrays, dictionaries, tabular data, and matrices. These inputs are mapped to standard or user-defined anatomical resources.

The package builds on established scientific Python libraries. It uses `nibabel` for neuroimaging file I/O [@Brett:2024], `numpy` for array operations [@Harris:2020], `pandas` for tabular and labelled data inputs [@McKinney:2010], `scipy` for interpolation and sparse matrix operations [@Virtanen:2020], `pooch` for resource retrieval and caching [@Uieda:2020], `scikit-image` for surface-mesh extraction from volumetric masks [@vanDerWalt:2014], `pyvista` for 3D mesh representation and rendering [@Sullivan:2019], `trame` for interactive browser-based visualization [@Jourdain:2025], and `matplotlib` for static figure composition [@Hunter:2007]. By combining these tools behind a compact plotting API, `yabplot` handles data mapping, mesh creation, camera placement, lighting, and rendering while retaining compatibility with standard Python plotting workflows.

![Overview of `yabplot` functionality. The package provides utilities for resource access, atlas construction, volume projection, and mesh handling, together with high-level plotting functions for cortical parcellations, vertex-wise surface maps, subcortical structures, voxel-wise volumes, white matter tracts, and connectome graphs.](figures/overview_joss.pdf)

# Statement of need

Researchers working with neuroimaging data often need to communicate results across several anatomical representations [@Chopra:2023]. The target users of `yabplot` are neuroimaging researchers and students who want to generate consistent 3D figures directly in Python-based data analysis workflows.

Producing such figures can be difficult when each representation requires a different application or plotting library. Switching between tools can require data conversion, manual screenshot-based workflows, and repeated adjustment of lighting, camera angles, colormaps, and output formatting. `yabplot` addresses this need by providing a unified, scriptable interface for common 3D neuroimaging visualizations, with consistent styling and output handling across modalities.

# State of the field

The neuroimaging visualization ecosystem includes many mature and widely used tools. Connectome Workbench supports surface-based visualization and Human Connectome Project resources [@Marcus:2011], MRtrix3 provides diffusion MRI processing and tractography visualization [@Tournier:2019], and BrainNet Viewer supports graph-based visualization of human connectomes [@Xia:2013]. In Python, Nilearn provides accessible statistical neuroimaging visualization, especially for slice-based, glass-brain, and machine-learning workflows [@Abraham:2014], while BrainSpace supports surface-based visualization in the context of macroscale gradients and connectomics [@VosdeWael:2020].

These tools cover important parts of the visualization workflow, but they are often specialized for one representation or analysis domain. Rather than replacing these packages, `yabplot` provides a complementary layer focused on consistent, publication-oriented 3D figure generation across cortical, subcortical, tractography, voxel-wise, and connectome data. This unified focus is the main reason for building a new package rather than contributing a modality-specific feature to an existing tool.

# Software design

`yabplot` is organized around high-level plotting functions that share common utilities for data mapping, mesh handling, camera presets, color handling, background anatomy, and output generation. This design trades some low-level rendering flexibility for consistency and ease of use across modalities. Users provide data values and an anatomical target, choose views and visual styling, and receive a rendered static figure or interactive object.

A further design goal is to separate the Python package from large anatomical resources. Supported atlases, meshes, and tract resources are retrieved on demand and cached locally, keeping the core installation lightweight while making resource use explicit. The package also supports custom user-defined anatomical resources through atlas-building and mesh-construction utilities, allowing users to adapt the workflow to study-specific parcellations, segmentations, and tractography datasets rather than being limited to bundled templates.

# Research impact

`yabplot` is designed to support reproducible figure generation in neuroimaging projects that combine several anatomical representations. The package is publicly available on PyPI, documented with example workflows, archived with a Zenodo DOI, and tested through continuous integration. Early user feedback has informed support for Matplotlib-compatible customization, custom atlas generation, and expanded plotting functionality.

Beyond standalone figure generation, `yabplot` is already used by the external `subcortex_visualization` package [@Bryant:2026], where its atlas-building and rendering utilities support a semi-automated workflow for converting volumetric segmentations into SVG-based custom atlas visualizations. These community-readiness signals provide a foundation for near-term use in neuroimaging studies requiring consistent 3D visualization across modalities.

# AI usage disclosure

Generative AI tools, including ChatGPT and Gemini, were used during the development of `yabplot` and preparation of this manuscript. They were used for code refactoring suggestions, debugging assistance, documentation drafting, and manuscript outlining. The author reviewed, edited, and verified all AI-assisted code, documentation, and manuscript text, and takes full responsibility for the submitted software and manuscript.

# Acknowledgements

The author would like to thank the early contributors, including Anthony J. Barrows, Jannis Denecke, and Anthony Gagnon, for their valuable code improvements and feedback during the development of this package. `yabplot` relies on the work of the broader neuroimaging and scientific Python communities, including the developers and maintainers of `nibabel`, `numpy`, `scipy`, `pandas`, `pyvista`, `trame`, `matplotlib`, `pooch`, `scikit-image`, and the atlas resources supported by the package. Users should cite the original publications for any atlases used in their analyses.

# References
