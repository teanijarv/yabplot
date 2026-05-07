import os
import gc
import re
import numpy as np
import pandas as pd
import nibabel as nib
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgba

from .data import (
    get_surface_paths, _resolve_resource_path, _find_cortical_files,
    _find_subcortical_files, _find_tract_files, get_atlas_regions,
    get_available_resources
)

from .utils import (
    load_gii, load_gii2pv, prep_data,
    generate_distinct_colors, parse_lut
)

from .mesh import (
    map_values_to_surface, get_puzzle_pieces, apply_internal_blur,
    apply_dilation, get_smooth_mask, lines_from_streamlines,
    make_cortical_mesh, load_bmesh, extract_polydata, get_region_boundaries
)

from .scene import (
    get_view_configs, prepare_plotter, setup_plotter, add_context_to_view,
    set_camera, finalize_plot, get_shading_preset, add_colorbars
)


def _build_contour_layers(draw_contours, contour_regions,
                           lh_v, lh_f, rh_v, rh_f,
                           tar_labels, lh_vals_raw, rh_vals_raw,
                           lut_ids, lut_names, n_lh):
    """Build a list of (lh_mesh, rh_mesh, kwargs) contour render layers."""
    base_kwargs = {'color': 'black', 'line_width': 2.0, 'opacity': 1.0, 'include_nan': True}
    if isinstance(draw_contours, dict):
        base_kwargs.update(draw_contours)
    base_include_nan = base_kwargs.pop('include_nan')

    # name → label-ID lookup (lut_names is a dense list indexed by label ID)
    name_to_id = {name: lid for lid, name in enumerate(lut_names)
                  if name and name != 'Unknown'}

    # resolve contour_regions → groups: list of (region_ids_set_or_None, kwargs, include_nan)
    if contour_regions is None:
        groups = [(None, base_kwargs, base_include_nan)]

    elif isinstance(contour_regions, dict):
        from collections import defaultdict
        kwargs_groups = defaultdict(list)
        for rname, rkwargs in contour_regions.items():
            merged = {**base_kwargs, **rkwargs}
            inc_nan = merged.pop('include_nan', base_include_nan)
            key = (tuple(sorted(merged.items())), inc_nan)
            rid = name_to_id.get(rname)
            if rid is not None:
                kwargs_groups[key].append(rid)
        groups = [(set(ids), dict(kw), inc_nan) for (kw, inc_nan), ids in kwargs_groups.items()]

    else:
        # list of names or boolean mask
        cr = np.asarray(contour_regions)
        if cr.dtype == bool:
            region_ids = {int(lut_ids[i]) for i, m in enumerate(cr) if m and i < len(lut_ids)}
        else:
            region_ids = {name_to_id[n] for n in cr if n in name_to_id}
        groups = [(region_ids, base_kwargs, base_include_nan)]

    layers = []
    for region_ids_set, kwargs, include_nan in groups:
        lh_c = get_region_boundaries(lh_v, lh_f, tar_labels[:n_lh],
                                      values=lh_vals_raw, include_nan=include_nan,
                                      region_ids=region_ids_set)
        rh_c = get_region_boundaries(rh_v, rh_f, tar_labels[n_lh:],
                                      values=rh_vals_raw, include_nan=include_nan,
                                      region_ids=region_ids_set)
        if lh_c is not None or rh_c is not None:
            layers.append((lh_c, rh_c, kwargs))
    return layers


def _render_cortical_views(lh_v, lh_f, lh_vals, rh_v, rh_f, rh_vals, is_cat,
                           ax, cbar_kwargs,
                           views, layout, figsize, cmap, vminmax, nan_color,
                           style, zoom, proc_vertices, display_type, export_path,
                           lut_colors=None, max_id=None,
                           contour_layers=None):
    """Internal helper to render cortical data."""

    # setup colors and vminmax
    n_colors = 256
    if is_cat:
        _lut_colors = lut_colors.copy()
        _lut_colors[0] = nan_color
        cmap = ListedColormap(_lut_colors)
        n_colors = len(_lut_colors)
        vmin, vmax = 0, max_id
    else:
        all_vals = np.concatenate([lh_vals, rh_vals])
        vmin = vminmax[0] if vminmax[0] is not None else np.nanmin(all_vals)
        vmax = vminmax[1] if vminmax[1] is not None else np.nanmax(all_vals)

    # process vertices
    results = []
    for v, f, raw in [(lh_v, lh_f, lh_vals), (rh_v, rh_f, rh_vals)]:
        if proc_vertices == 'sharp':
            base, pieces = get_puzzle_pieces(v, f, raw)
            results.append((base, pieces))
        else:
            v_proc = apply_internal_blur(f, raw, iterations=3, weight=0.3) if proc_vertices == 'blur' else raw
            dilated = apply_dilation(f, v_proc, iterations=4)
            o_guide = get_smooth_mask(f, np.where(np.isnan(raw), 0.0, 1.0), iterations=4)

            mesh = make_cortical_mesh(v, f, dilated)
            mesh['Slice_Mask'] = o_guide
            data_p = mesh.clip_scalar(scalars='Slice_Mask', value=0.5, invert=False)
            base_p = mesh.clip_scalar(scalars='Slice_Mask', value=0.5, invert=True)
            if base_p.n_points > 0: base_p['Data'] = np.full(base_p.n_points, np.nan)
            results.append((base_p, [data_p]))
    (lh_base, lh_parts), (rh_base, rh_parts) = results

    # plotter setup
    sel_views = get_view_configs(views)
    ax, display_type, figsize = prepare_plotter(ax, display_type, sel_views, layout, figsize)
    plotter, ncols, nrows = setup_plotter(sel_views, layout, figsize, display_type)
    shading_params = get_shading_preset(style)
    scalar_bar_mapper = None

    for i, (name, cfg) in enumerate(sel_views.items()):
        plotter.subplot(i // ncols, i % ncols)

        view_bases = []
        view_pieces = []
        if cfg['side'] in ['L', 'both']:
            if lh_base.n_points > 0: view_bases.append(lh_base)
            view_pieces.extend(lh_parts)
        if cfg['side'] in ['R', 'both']:
            if rh_base.n_points > 0: view_bases.append(rh_base)
            view_pieces.extend(rh_parts)

        # brain meshes
        for b_mesh in view_bases:
            plotter.add_mesh(b_mesh, color=nan_color, smooth_shading=True, **shading_params)

        # data vertices
        for p_mesh in view_pieces:
            if p_mesh.n_points == 0: continue
            interp = (proc_vertices == 'blur')

            actor = plotter.add_mesh(
                p_mesh, scalars='Data', cmap=cmap, clim=(vmin, vmax),
                n_colors=n_colors, nan_color=nan_color, show_scalar_bar=False,
                smooth_shading=True, interpolate_before_map=interp, **shading_params
            )
            if scalar_bar_mapper is None: scalar_bar_mapper = actor.mapper

        # draw region boundary contours if requested
        if contour_layers:
            for lh_c, rh_c, ckw in contour_layers:
                for c_mesh, side_key in [(lh_c, 'L'), (rh_c, 'R')]:
                    if c_mesh is None:
                        continue
                    if cfg['side'] in [side_key, 'both']:
                        plotter.add_mesh(c_mesh, **ckw)

        set_camera(plotter, cfg, zoom=zoom)
        plotter.hide_axes()

    cbar_info = []
    if not is_cat and scalar_bar_mapper:
        if display_type != 'matplotlib':
            add_colorbars(plotter, [scalar_bar_mapper], [''], nrows, figsize)
        else:
            cbar_info.append({'cmap': cmap, 'vminmax': [vmin, vmax]})

    return finalize_plot(plotter, export_path, display_type, ax=ax, cbar_info=cbar_info, cbar_kwargs=cbar_kwargs)



### PLOT FOR ATLAS-BASED CORTICAL DATA ###

def plot_cortical(data=None, atlas=None, custom_atlas_path=None, ax=None, cbar_kwargs=None, views=None, layout=None,
                  bmesh='midthickness', figsize=None, cmap='coolwarm', vminmax=[None, None],
                  nan_color=(1.0, 1.0, 1.0), style='default', zoom=1.2, proc_vertices=None,
                  display_type='matplotlib', export_path=None, draw_contours=False,
                  contour_regions=None):
    """
    Visualize data on the cortical surface using a specified atlas.

    This function maps scalar values to cortical regions (parcellations) on a standard
    surface mesh (Conte69). It supports both pre-existing atlases and custom local atlases.

    Parameters
    ----------
    data : dict, list, numpy.ndarray, optional
        Data to map onto the cortex.
        If dict: Keys must match region names in the atlas (see `yabplot.get_atlas_regions`).
        If array/list: Must match the exact length and order of the atlas regions.
        If None: The atlas is plotted with categorical colors (one color per region).
    atlas : str, optional
        Name of the standard atlas to use (e.g., 'schaefer_100',
        see 'yabplot.get_available_resources' for more).
        Defaults to 'aparc' if neither atlas nor custom_atlas_path is provided.
    custom_atlas_path : str, optional
        Path to a local directory containing custom atlas files. The directory must
        contain a CSV mapping regions to vertices and a LUT text file. If provided, `atlas` is ignored.
    views : list of str, optional
        Views to display. Can be a list of presets ('left_lateral', 'right_medial', etc.)
        or a dictionary of camera configurations. Defaults to all views.
    layout : tuple (rows, cols), optional
        Grid layout for subplots. If None, automatically calculated based on the number of views.
    bmesh : str
        Name of the background context brain mesh (e.g., 'midthickness', 'white', 'swm', etc).
        Default is 'midthickness'.
    figsize : tuple (width, height), optional
        Window size in inches. If None, automatically calculated based on the number of views and layout.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap for continuous data. Ignored if `data` is None. Default is 'RdYlBu_r'.
    vminmax : list [min, max], optional
        Manual lower and upper bounds for the colormap. If [None, None],
        bounds are inferred from the data range.
    nan_color : tuple or str, optional
        Color for regions with missing (NaN) data or the medial wall. Default is white.
    style : str, optional
        Lighting preset ('default', 'matte', 'glossy', 'sculpted', 'flat').
    zoom : float, optional
        Camera zoom level. >1.0 zooms in, <1.0 zooms out. Default is 1.2.
    proc_vertices : str or None, optional
        Whether to process the vertices edges according to geometry of bmesh.
        Set to None to not perform (default).
        'blur': Applies simple blurring between different color vertices (low performance impact).
        'sharp': Applies sharpening of the resolution of different color vertices (high performance impact).
    display_type : {'matplotlib', 'interactive', 'pyvista', 'object'}, optional
        'matplotlib': returns a matplotlib figure and axis (default).
        'interactive': opens an interactive trame viewer in the browser.
        'pyvista': returns a static jupyter widget (legacy behavior).
        'object': returns the raw pyvista plotter object.
    export_path : str, optional
        If provided, saves the final figure to this path (e.g., 'figure.png').
    draw_contours : bool or dict, optional
        Draw boundary lines between atlas regions on the surface.
        False (default): no contours.
        True: draw with default style (black lines, width 2, fully opaque).
        dict: override any of the defaults. Supported keys (beyond PyVista
        ``add_mesh`` kwargs):

        - ``color`` (str/tuple): line color. Default ``'black'``.
        - ``line_width`` (float): line width in screen pixels. Default ``2.0``.
        - ``opacity`` (float): line opacity 0–1. Default ``1.0``.
        - ``include_nan`` (bool): if False, only draw edges where at least one
          side has a non-NaN value — outlines regions that have data, including
          their boundary with NaN regions, but skips borders between two NaN
          regions. Default ``True``.
    contour_regions : None, list, numpy.ndarray, or dict, optional
        Restrict contours to specific atlas regions (requires ``draw_contours``
        to be enabled). If None (default), contours are drawn for all regions.

        - ``list`` of region name strings: draw contours only for those regions,
          using the global style from ``draw_contours``.
        - ``numpy.ndarray`` (bool, length = number of atlas regions in LUT order):
          draw contours for regions where the mask is True.
        - ``dict`` mapping region name → style-kwargs dict: draw contours only
          for the listed regions; each region's style is the global
          ``draw_contours`` defaults merged with its per-region overrides.
          Supported per-region keys: same as ``draw_contours`` dict keys
          (``color``, ``line_width``, ``opacity``, ``include_nan``).

    Returns
    -------
    matplotlib.axes.Axes or pyvista.Plotter or IPython.display.DisplayObject
        returns based on display_type:
        - 'matplotlib': returns a matplotlib axes object.
        - 'interactive': returns a trame browser viewer.
        - 'pyvista': returns a static jupyter widget.
        - 'object': returns the raw pyvista plotter.
    """

    # atlas and categorical check
    if atlas is None and custom_atlas_path is None:
        atlas = 'aparc'
    is_cat = (data is None)

    # load brain mesh
    b_lh_path, b_rh_path = get_surface_paths(bmesh, 'bmesh')
    lh_v, lh_f = load_gii(b_lh_path)
    rh_v, rh_f = load_gii(b_rh_path)

    # resolve atlas
    atlas_dir = _resolve_resource_path(atlas, 'cortical', custom_path=custom_atlas_path)
    check_name = None if custom_atlas_path else atlas
    csv_path, lut_path = _find_cortical_files(atlas_dir, strict_name=check_name)

    # load mapping data
    tar_labels = np.loadtxt(csv_path, dtype=int)
    lut_ids, lut_colors, lut_names, max_id = parse_lut(lut_path)

    # map data
    all_vals = map_values_to_surface(data, tar_labels, lut_ids, lut_names)
    lh_vals_raw = all_vals[:len(lh_v)]
    rh_vals_raw = all_vals[len(lh_v):]

    # compute region boundary contours
    contour_layers = None
    if draw_contours:
        n_lh = len(lh_v)
        contour_layers = _build_contour_layers(
            draw_contours, contour_regions,
            lh_v, lh_f, rh_v, rh_f,
            tar_labels, lh_vals_raw, rh_vals_raw,
            lut_ids, lut_names, n_lh
        )

    # render
    return _render_cortical_views(
        lh_v, lh_f, lh_vals_raw, rh_v, rh_f, rh_vals_raw, is_cat, ax, cbar_kwargs,
        views, layout, figsize, cmap, vminmax, nan_color, style,
        zoom, proc_vertices, display_type, export_path, lut_colors, max_id,
        contour_layers
    )



### PLOT FOR VERTEX-WISE CORTICAL DATA ###

def plot_vertexwise(lh, rh, scalars='Data', ax=None, cbar_kwargs=None, views=None, layout=None, figsize=None,
                    cmap='coolwarm', vminmax=[None, None],
                    nan_color=(1.0, 1.0, 1.0), style='default', zoom=1.2,
                    proc_vertices=None, display_type='matplotlib', export_path=None):
    """
    Visualize arbitrary per-vertex scalar data on a user-supplied brain mesh.

    Unlike `plot_cortical`, this function requires no atlas. The user provides
    PyVista PolyData meshes with per-vertex scalar data stored under the key specified
    by `scalars`.

    Parameters
    ----------
    lh : pyvista.PolyData
        Left hemisphere mesh containing a (N,) float array under ``lh[scalars]``.
    rh : pyvista.PolyData
        Right hemisphere mesh containing a (N,) float array under ``rh[scalars]``.
    scalars : str, optional
        The string key corresponding to the scalar data array in the PyVista
        point data dictionary. Default is 'Data'.
    views : list of str, optional
        Can be a list of presets ('left_lateral', 'right_medial', etc.)
        or a dictionary of camera configurations. Defaults to all views.
    layout : tuple (rows, cols), optional
        Grid layout for subplots. If None, auto-calculated.
    figsize : tuple (width, height), optional
        Window size in inches. If None, automatically calculated based on the number of views and layout.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap. Default is 'coolwarm'.
    vminmax : list [min, max], optional
        Colormap bounds. If [None, None], inferred from data range.
    nan_color : tuple or str, optional
        Color for NaN vertices. Default is white.
    style : str, optional
        Lighting preset ('default', 'matte', 'glossy', 'sculpted', 'flat').
    zoom : float, optional
        Camera zoom level. Default is 1.2.
    proc_vertices : str or None, optional
        Vertex processing mode: None, 'blur', or 'sharp'.
    display_type : {'matplotlib', 'interactive', 'pyvista', 'object'}, optional
        'matplotlib': returns a matplotlib figure and axis (default).
        'interactive': opens an interactive trame viewer in the browser.
        'pyvista': returns a static jupyter widget (legacy behavior).
        'object': returns the raw pyvista plotter object.
    export_path : str, optional
        If provided, saves the figure to this path.

    Returns
    -------
    matplotlib.axes.Axes or pyvista.Plotter or IPython.display.DisplayObject
        returns based on display_type:
        - 'matplotlib': returns a matplotlib axes object.
        - 'interactive': returns a trame browser viewer.
        - 'pyvista': returns a static jupyter widget.
        - 'object': returns the raw pyvista plotter.

    See Also
    --------
    yabplot.mesh.load_vertexwise_mesh

    Examples
    --------
    >>> from yabplot.mesh import load_vertexwise_mesh
    >>> lh, rh = load_vertexwise_mesh(
    ...     fsaverage.pial_left, fsaverage.pial_right,
    ...     d_values_lh, d_values_rh
    ... )
    >>> # If your data was injected under the default 'Data' key
    >>> plot_vertexwise(lh, rh, views=['left_lateral', 'right_lateral'])
    >>>
    >>> # If your data was injected under a custom key
    >>> lh['thickness'] = lh_thick_array
    >>> rh['thickness'] = rh_thick_array
    >>> plot_vertexwise(lh, rh, scalars='thickness', cmap='inferno')
    """

    # extract v, f, raw from PyVista meshes
    lh_v, lh_f = extract_polydata(lh)
    lh_vals_raw = lh[scalars]
    rh_v, rh_f = extract_polydata(rh)
    rh_vals_raw = rh[scalars]

    # render
    return _render_cortical_views(
        lh_v, lh_f, lh_vals_raw, rh_v, rh_f, rh_vals_raw, False, ax, cbar_kwargs,
        views, layout, figsize, cmap, vminmax, nan_color, style,
        zoom, proc_vertices, display_type, export_path
    )



### PLOT FOR ATLAS-BASED SUBCORTICAL DATA ###

def plot_subcortical(data=None, atlas=None, custom_atlas_path=None, ax=None, cbar_kwargs=None, views=None, layout=None,
                     figsize=None, cmap='coolwarm', vminmax=[None, None], nan_color='#cccccc',
                     nan_alpha=1.0, style='default', bmesh='midthickness',
                     bmesh_alpha=0.2, bmesh_color='lightgray', zoom=1.2, display_type='matplotlib',
                     export_path=None, custom_atlas_proc=dict(smooth_i=15, smooth_f=0.6)):
    """
    Visualize data on the subcortical structures using a specified atlas.

    Renders volumetric structures as 3D meshes. Supports pre-existing atlases and
    on-the-fly conversion of GIfTI surfaces to smooth meshes for custom atlases.

    Parameters
    ----------
    data : dict, list, numpy.ndarray, pandas.Series, pandas.DataFrame, optional
        Scalar values for each subcortical region.
        If dict/pd.Series/pd.DataFrame: Values according to region names.
        If array/list: Must strictly match the sorted order of regions in the atlas.
    atlas : str, optional
        Name of the standard atlas to use (e.g., 'musus_100',
        see 'yabplot.get_available_resources' for more).
        Defaults to 'aseg' if neither atlas nor custom_atlas_path is provided.
    custom_atlas_path : str, optional
        Path to a local directory containing .vtk or .gii mesh files for each region.
    views : list of str, optional
        Views to display. Can be a list of presets ('left_lateral', 'right_medial', etc.)
        or a dictionary of camera configurations. Defaults to all views.
    layout : tuple (rows, cols), optional
        Grid layout for subplots. If None, automatically calculated based on the number of views.
    figsize : tuple (width, height), optional
        Window size in inches. If None, automatically calculated based on the number of views and layout.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap for continuous data. Ignored if `data` is None. Default is 'coolwarm'.
    vminmax : list [min, max], optional
        Manual lower and upper bounds for the colormap. If [None, None],
        bounds are inferred from the data range.
    nan_color : str or tuple, optional
        Color for regions with no data (NaN). Default is light grey '#cccccc'.
    nan_alpha : float, optional
        Opacity (0.0 to 1.0) for regions with no data. Set to 0.0 to hide them.
    style : str, optional
        Lighting preset ('default', 'matte', 'glossy', 'sculpted', 'flat').
    bmesh : pyvista.PolyData or dict, optional
        Configure background context brain mesh. Accepts a string
        (e.g., 'midthickness', 'white', 'swm', etc), single PolyData (used for both hemispheres)
        or a dict with 'L'/'R' keys. Default is 'midthickness'.
    bmesh_alpha : float, optional
        Opacity of the context brain mesh. Default is 0.2.
    bmesh_color : str, optional
        Color of the context brain mesh.
    zoom : float, optional
        Camera zoom level. >1.0 zooms in, <1.0 zooms out. Default is 1.2.
    display_type : {'matplotlib', 'interactive', 'pyvista', 'object'}, optional
        'matplotlib': returns a matplotlib figure and axis (default).
        'interactive': opens an interactive trame viewer in the browser.
        'pyvista': returns a static jupyter widget (legacy behavior).
        'object': returns the raw pyvista plotter object.
    export_path : str, optional
        If provided, saves the final figure to this path (e.g., 'figure.png').
    custom_atlas_proc : dict, optional
        Parameters for processing custom GIfTI files.
        Keys: 'smooth_i' (iterations) and 'smooth_f' (relaxation factor).
        Default is {'smooth_i': 15, 'smooth_f': 0.6}.

    Returns
    -------
    matplotlib.axes.Axes or pyvista.Plotter or IPython.display.DisplayObject
        returns based on display_type:
        - 'matplotlib': returns a matplotlib axes object.
        - 'interactive': returns a trame browser viewer.
        - 'pyvista': returns a static jupyter widget.
        - 'object': returns the raw pyvista plotter.
    """

    # defaults
    if atlas is None and custom_atlas_path is None:
        atlas = 'aseg'

    # load context brain mesh (if requested) or accept mesh directly
    ctx_meshes = load_bmesh(bmesh)

    # load regional atlas meshes
    # resolve atlas path (either download or custom directory)
    atlas_dir = _resolve_resource_path(atlas, 'subcortical', custom_path=custom_atlas_path)

    # locate mesh files, returns dict: {'Left_Thalamus': '/path/to/Left_Thalamus.vtk', ...}
    file_map = _find_subcortical_files(atlas_dir)
    rmesh_names = get_atlas_regions(atlas, 'subcortical', custom_atlas_path)

    # load meshes from cache or disk
    meshes = {}
    cache_key = 'custom' if custom_atlas_path else atlas
    for name, fpath in file_map.items():
        mesh = _retrieve_static_mesh('subcortical', cache_key, name, fpath, **custom_atlas_proc)
        if mesh:
            meshes[name] = mesh

    # prepare colors and map data
    if data is not None:
        d_data = prep_data(data, rmesh_names, atlas, 'subcortical')
        valid_vals = [v for v in d_data.values() if pd.notna(v)]
        vmin = vminmax[0] if vminmax[0] is not None else (min(valid_vals) if valid_vals else 0)
        vmax = vminmax[1] if vminmax[1] is not None else (max(valid_vals) if valid_vals else 1)
        c_vlim = [vmin, vmax]
    else:
        colors = generate_distinct_colors(len(rmesh_names), seed=42)
        d_atlas_colors = {name: color for name, color in zip(rmesh_names, colors)}
        c_vlim = [0, 1]

    # setup plotter
    sel_views = get_view_configs(views)
    ax, display_type, figsize = prepare_plotter(ax, display_type, sel_views, layout, figsize)

    needs_bottom = (data is not None)
    plotter, ncols, nrows = setup_plotter(sel_views, layout, figsize, display_type,
                                           needs_bottom_row=needs_bottom)


    # get shading parameters from style
    shading_params = get_shading_preset(style)
    scalar_bar_mapper = None

    # pre-calculate side tokens for all meshes to avoid regex in loops
    side_info = {n: _get_side_tokens(n) for n in meshes.keys()}

    # plotting loop
    for i, (view_name, cfg) in enumerate(sel_views.items()):
        plotter.subplot(i // ncols, i % ncols)

        # add context (uses style kwargs for consistent lighting)
        add_context_to_view(plotter, ctx_meshes, cfg['side'], bmesh_alpha, bmesh_color,
                            **shading_params)

        # add regions
        for name, mesh in meshes.items():
            # side filtering using pre-calculated tokens
            is_left, is_right = side_info[name]
            if cfg['side'] == 'L' and is_right and not is_left: continue
            if cfg['side'] == 'R' and is_left and not is_right: continue

            # determine properties for this mesh
            props = shading_params.copy()

            if data is not None:
                val = d_data.get(name, np.nan) if pd.notna(d_data.get(name)) else np.nan
                has_val = not np.isnan(val)

                mesh['Data'] = np.full(mesh.n_points, val)

                props.update({
                    'scalars': 'Data', 'cmap': cmap, 'clim': c_vlim,
                    'nan_color': nan_color, 'opacity': 1.0 if has_val else nan_alpha,
                    'show_scalar_bar': False
                })
            else:
                color = d_atlas_colors[name]
                props.update({'color': color, 'opacity': 1.0})

            actor = plotter.add_mesh(mesh, **props)

            if data is not None and scalar_bar_mapper is None and 'scalars' in props:
                 scalar_bar_mapper = actor.mapper

        set_camera(plotter, cfg, zoom=zoom)
        plotter.hide_axes()

    # colorbar
    cbar_info = []
    if needs_bottom and scalar_bar_mapper:
        if display_type != 'matplotlib':
            add_colorbars(plotter, [scalar_bar_mapper], [''], nrows, figsize)
        else:
            cbar_info.append({'cmap': cmap, 'vminmax': c_vlim})

    return finalize_plot(plotter, export_path, display_type, ax=ax, cbar_info=cbar_info, cbar_kwargs=cbar_kwargs)



### PLOT FOR ATLAS-BASED WHITE MATTER TRACT DATA ###

from collections import OrderedDict

# global cache for static geometry (subcortical/tracts) with lru logic
_STATIC_CACHE = OrderedDict()
_STATIC_CACHE_LIMIT = 100 # max individual meshes to keep in ram

def clear_cache():
    """manually clears the global geometry cache to free ram."""
    global _STATIC_CACHE
    _STATIC_CACHE.clear()
    gc.collect()
    print("geometry cache cleared.")

def _get_side_tokens(name):
    """pre-calculates side identity for a given mesh name."""
    tokens = set(re.split(r'[^a-z0-9]+', name.lower()))
    is_left = any(x in tokens for x in ['left', 'l', 'lh'])
    is_right = any(x in tokens for x in ['right', 'r', 'rh'])
    return is_left, is_right

def _retrieve_static_mesh(category, atlas_key, name, fpath, **kwargs):
    """retrieves a mesh from lru cache or loads from disk."""
    global _STATIC_CACHE
    cache_id = f"{category}_{atlas_key}_{name}"

    # check ram cache and move to end (mru)
    if cache_id in _STATIC_CACHE:
        _STATIC_CACHE.move_to_end(cache_id)
        return _STATIC_CACHE[cache_id]

    # load from disk
    try:
        if category == 'tracts':
            tractogram = nib.streamlines.load(fpath)
            points, lines, tangents = lines_from_streamlines(tractogram.streamlines)
            if len(points) == 0: return None
            mesh = pv.PolyData(points, lines=lines)
            mesh.point_data['tangents'] = np.abs(tangents)
        else:
            # subcortical
            if fpath.endswith('.vtk'):
                mesh = pv.read(fpath)
            elif fpath.endswith('.gii'):
                mesh = load_gii2pv(fpath, **kwargs)
            else:
                return None

        # store in lru cache
        _STATIC_CACHE[cache_id] = mesh
        if len(_STATIC_CACHE) > _STATIC_CACHE_LIMIT:
            _STATIC_CACHE.popitem(last=False) # drop oldest

        return mesh

    except Exception as e:
        print(f"failed to load {category} mesh {name}: {e}")
        return None

def plot_tracts(data=None, atlas=None, custom_atlas_path=None, ax=None, cbar_kwargs=None, views=None, layout=None,
                figsize=None, cmap='coolwarm', alpha=1.0, vminmax=[None, None],
                nan_color='#BDBDBD', nan_alpha=1.0, style='default',
                bmesh='midthickness', bmesh_alpha=0.2, bmesh_color='lightgray',
                zoom=1.2, orientation_coloring=False, display_type='matplotlib',
                tract_kwargs=dict(render_lines_as_tubes=True, line_width=1.2),
                export_path=None):
    """
    Visualize data on the white matter tractography bundles using a specified atlas.

    Renders streamlines from .trk files. Can color tracts by scalar values,
    categorically, or by local fiber orientation.

    Parameters
    ----------
    data : dict, list, numpy.ndarray, pandas.Series, pandas.DataFrame, optional
        Scalar values for each tract, or mrtrix3 derived .tsf file path for each tract.
        If dict: Keys must match tract names.
        If array/list: Must strictly match the sorted list of tracts in the atlas.
        If None: Tracts are colored by category (distinct colors) or orientation.
    atlas : str, optional
        Name of the standard tract atlas (e.g., 'hcp1065_small',
        see 'yabplot.get_available_resources' for more).
        Defaults to 'xtract_tiny'.
    custom_atlas_path : str, optional
        Path to a local directory containing .trk files for each tract.
    views : list of str, optional
        Views to display. Can be a list of presets ('left_lateral', 'right_medial', etc.)
        or a dictionary of camera configurations. Defaults to all views.
    layout : tuple (rows, cols), optional
        Grid layout for subplots. If None, automatically calculated based on the number of views.
    figsize : tuple (width, height), optional
        Window size in inches. If None, automatically calculated based on the number of views and layout.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap for continuous data. Ignored if `data` is None. Default is 'coolwarm'.
    alpha : float, optional
        Opacity of the tracts (0.0 to 1.0).
    vminmax : list [min, max], optional
        Manual lower and upper bounds for the colormap. If [None, None],
        bounds are inferred from the data range.
    nan_color : str, optional
        Color for tracts with missing data (NaN). Default is grey '#BDBDBD'.
    nan_alpha : float, optional
        Opacity (0.0 to 1.0) for regions with no data. Set to 0.0 to hide them.
    style : str, optional
        Lighting preset ('default', 'matte', 'glossy', 'sculpted', 'flat').
    bmesh : pyvista.PolyData or dict, optional
        Configure background context brain mesh. Accepts a string
        (e.g., 'midthickness', 'white', 'swm', etc), single PolyData (used for both hemispheres)
        or a dict with 'L'/'R' keys. Default is 'midthickness'.
    bmesh_alpha : float, optional
        Opacity of the context brain mesh. Default is 0.2.
    bmesh_color : str, optional
        Color of the context brain mesh.
    zoom : float, optional
        Camera zoom level. >1.0 zooms in, <1.0 zooms out. Default is 1.2.
    orientation_coloring : bool, optional
        If True, ignores `data` and colors fibers based on their local directional
        orientation (Red=L/R, Green=A/P, Blue=S/I).
    tract_kwargs : dict, optional
        Additional arguments passed to PyVista's `add_mesh`.
        Default configures tubes: `{'render_lines_as_tubes': True, 'line_width': 1.2}`.
    display_type : {'matplotlib', 'interactive', 'pyvista', 'object'}, optional
        'matplotlib': returns a matplotlib figure and axis (default).
        'interactive': opens an interactive trame viewer in the browser.
        'pyvista': returns a static jupyter widget (legacy behavior).
        'object': returns the raw pyvista plotter object.
    export_path : str, optional
        If provided, saves the final figure to this path (e.g., 'figure.png').

    Returns
    -------
    matplotlib.axes.Axes or pyvista.Plotter or IPython.display.DisplayObject
        returns based on display_type:
        - 'matplotlib': returns a matplotlib axes object.
        - 'interactive': returns a trame browser viewer.
        - 'pyvista': returns a static jupyter widget.
        - 'object': returns the raw pyvista plotter.
    """

    # defaults
    if atlas is None and custom_atlas_path is None:
        atlas = 'xtract_tiny'

    # resolve atlas path (either download or custom directory)
    atlas_dir = _resolve_resource_path(atlas, 'tracts', custom_path=custom_atlas_path)

    # locate tract files, returns dict eg {'CST_L': '/path/to/CST_L.trk', ...}
    file_map = _find_tract_files(atlas_dir)
    tract_names = get_atlas_regions(atlas, 'tracts', custom_atlas_path)

    # prepare colors and map data
    if data is not None:
        d_data = prep_data(data, tract_names, atlas, 'tracts')
        all_vals = []
        for v in d_data.values():
            v_arr = np.atleast_1d(v)
            all_vals.append(v_arr[~np.isnan(v_arr)])

        if all_vals:
            valid_vals = np.concatenate(all_vals)
            vmin = vminmax[0] if vminmax[0] is not None else (np.min(valid_vals) if len(valid_vals) else 0)
            vmax = vminmax[1] if vminmax[1] is not None else (np.max(valid_vals) if len(valid_vals) else 1)
        else:
            vmin, vmax = 0, 1
        c_vlim = [vmin, vmax]
    # categorical/orientation mode
    else:
        colors = generate_distinct_colors(len(tract_names), seed=42)
        d_atlas_colors = {name: color for name, color in zip(tract_names, colors)}
        c_vlim = [0, 1]

    # load context brain mesh (if requested)
    ctx_meshes = load_bmesh(bmesh)

    # setup plotter
    sel_views = get_view_configs(views)
    ax, display_type, figsize = prepare_plotter(ax, display_type, sel_views, layout, figsize)

    needs_bottom = (data is not None and not orientation_coloring)
    plotter, ncols, nrows = setup_plotter(sel_views, layout, figsize, display_type,
                                           needs_bottom_row=needs_bottom)
    plotter.enable_depth_peeling(number_of_peels=10)
    plotter.enable_anti_aliasing('msaa') # smooth lines
    shading_params = get_shading_preset(style)
    scalar_bar_mapper = None

    # pre-calculate side tokens for all tracts to avoid regex in loops

    side_info = {n: _get_side_tokens(n) for n in tract_names}

    # plotting
    cache_key = 'custom' if custom_atlas_path else atlas
    for i, (view_name, cfg) in enumerate(sel_views.items()):
        plotter.subplot(i // ncols, i % ncols)

        # add context (passed shading params to context mesh)
        add_context_to_view(plotter, ctx_meshes, cfg['side'], bmesh_alpha, bmesh_color, **shading_params)

        # add tracts
        for name in tract_names:
            # optimization: early exit for hidden tracts
            has_value = False
            val = np.nan

            if data is not None and not orientation_coloring:
                # check data
                if name in d_data and d_data[name] is not None:
                    val = d_data[name]
                    if np.isscalar(val) and np.isnan(val):
                        has_value = False
                    elif not np.isscalar(val) and np.all(np.isnan(val)):
                        has_value = False
                    else:
                        has_value = True
                else:
                    has_value = False

                if not has_value and nan_alpha == 0:
                    continue

            # side filtering using pre-calculated tokens
            is_left, is_right = side_info[name]
            if cfg['side'] == 'L' and is_right and not is_left: continue
            if cfg['side'] == 'R' and is_left and not is_right: continue

            # load mesh from lru cache
            fpath = file_map.get(name)
            if not fpath: continue
            base_mesh = _retrieve_static_mesh('tracts', cache_key, name, fpath)
            if base_mesh is None: continue

            pv_mesh = base_mesh.copy(deep=False)
            # start with style presets, then override with tract_kwargs and dynamic props
            props = shading_params.copy()
            props.update(tract_kwargs)

            if orientation_coloring:
                pv_mesh['Data'] = pv_mesh.point_data['tangents']

                props.update({
                    'scalars': 'Data', 'rgb': True, 'opacity': alpha
                })

            elif data is not None:
                if np.isscalar(val):
                    pv_mesh['Data'] = np.full(pv_mesh.n_points, val)
                elif len(val) == 1:
                    pv_mesh['Data'] = np.full(pv_mesh.n_points, val[0])
                elif len(val) == pv_mesh.n_points:
                    pv_mesh['Data'] = val
                else:
                    raise ValueError(
                        f"Data shape mismatch for tract '{name}'. Must be a scalar "
                        f"or a 1D array matching the number of points. "
                        f"Array shape: {np.shape(val)}, mesh points: {pv_mesh.n_points}"
                    )

                current_opacity = alpha if has_value else nan_alpha

                props.update({
                    'scalars': 'Data', 'cmap': cmap, 'clim': c_vlim,
                    'nan_color': nan_color, 'opacity': current_opacity, 'show_scalar_bar': False
                })

            else:
                color = d_atlas_colors[name]
                props.update({
                    'color': color, 'opacity': alpha
                })

            actor = plotter.add_mesh(pv_mesh, **props)

            if data is not None and not orientation_coloring and scalar_bar_mapper is None and 'scalars' in props:
                scalar_bar_mapper = actor.mapper

        set_camera(plotter, cfg, zoom=zoom, distance=150)
        plotter.hide_axes()

    # colorbar
    cbar_info = []
    if needs_bottom and scalar_bar_mapper:
        if display_type != 'matplotlib':
            add_colorbars(plotter, [scalar_bar_mapper], [''], nrows, figsize)
        else:
            cbar_info.append({'cmap': cmap, 'vminmax': c_vlim})

    # finalize
    ret_val = finalize_plot(plotter, export_path, display_type, ax=ax, cbar_info=cbar_info, cbar_kwargs=cbar_kwargs)

    if display_type != 'interactive':
        del plotter
        gc.collect()

    return ret_val


### PLOT FOR ATLAS-BASED CONNECTOME DATA ###

def _extract_centroids(category, atlas, custom_atlas_path, bmesh_type):
    """calculates 3d spatial centers for all regions in a given atlas."""
    centroids, region_names, atlas_colors = {}, [], {}

    if category == 'subcortical':
        atlas_dir = _resolve_resource_path(atlas, 'subcortical', custom_path=custom_atlas_path)
        file_map = _find_subcortical_files(atlas_dir)
        region_names = sorted(list(file_map.keys()))

        # generate fallback distinct colors for subcortical regions
        gen_colors = generate_distinct_colors(len(region_names), seed=42)
        atlas_colors = {n: c for n, c in zip(region_names, gen_colors)}

        # calculate center of mass for each 3d volumetric mesh
        for name, fpath in file_map.items():
            mesh = pv.read(fpath) if fpath.endswith('.vtk') else load_gii2pv(fpath)
            centroids[name] = mesh.center_of_mass()

    elif category == 'cortical':
        atlas_dir = _resolve_resource_path(atlas, 'cortical', custom_path=custom_atlas_path)
        check_name = None if custom_atlas_path else atlas
        csv_path, lut_path = _find_cortical_files(atlas_dir, strict_name=check_name)
        labels = np.loadtxt(csv_path, dtype=int)

        # extract default atlas categorical colors using the standard parser
        _, lut_hex, lut_names, _ = parse_lut(lut_path)
        atlas_colors = {rname: rhex for rname, rhex in zip(lut_names, lut_hex)}

        # parse raw txt file to safely map region IDs to region names
        true_lut = {}
        with open(lut_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0].isdigit():
                    true_lut[int(parts[0])] = parts[1]

        # load raw geometric vertices to calculate spatial centers
        lh_path, rh_path = get_surface_paths(bmesh_type, 'bmesh')
        lh_v, _ = load_gii(lh_path)
        rh_v, _ = load_gii(rh_path)
        all_verts = np.vstack((lh_v, rh_v))

        # compute the mean coordinate (centroid) of all vertices belonging to each region
        for rid, rname in true_lut.items():
            if rname.lower() == 'unknown': continue

            mask = (labels == rid)
            if np.any(mask):
                centroids[rname] = all_verts[mask].mean(axis=0)
            else:
                # assign nan coordinates to tiny regions that fall completely between vertices
                import warnings
                warnings.warn(f"region '{rname}' (ID {rid}) has 0 vertices on this surface. it will be hidden.")
                centroids[rname] = np.array([np.nan, np.nan, np.nan])

            region_names.append(rname)

    return centroids, region_names, atlas_colors


def _parse_node_metrics(metric, mat, actual_thresh, directed, n_nodes, region_names):
    """
    parses the user's input into node scalar values.
    handles constants, internal graph metrics ('strength'), and external data arrays.
    """
    # 1. constant value (e.g., node_size=2.0)
    if isinstance(metric, (int, float)):
        return np.full(n_nodes, float(metric)), False, None

    # 2. internal graph metric
    elif isinstance(metric, str):
        if metric == 'strength':
            # calculate sum of surviving edge weights for each node (safely ignoring nans)
            with np.errstate(invalid='ignore'):
                m_t = np.where(np.abs(mat) > actual_thresh, np.abs(mat), 0)
            res = np.sum(m_t, axis=1) + (np.sum(m_t, axis=0) if directed else 0)
            return res, True, 'strength'
        else:
            raise ValueError(
                f"invalid metric string '{metric}'. the only supported internal graph "
                f"metric string is 'strength'. to use custom data, pass an array, dict, "
                f"list, or pandas object. for a constant size, pass an int or float."
            )

    # 3. external custom data (dict, series, dataframe, list, or array)
    elif isinstance(metric, dict):
        res = np.array([metric.get(n, 0.0) for n in region_names])
        return res, True, "data"
    elif isinstance(metric, pd.Series):
        res = metric.reindex(region_names).fillna(0).values if set(region_names).intersection(metric.index) else metric.values
        return res, True, "data"
    elif isinstance(metric, pd.DataFrame):
        res = metric.reindex(region_names).iloc[:, 0].fillna(0).values if set(region_names).intersection(metric.index) else metric.iloc[:, 0].values
        return res, True, "data"
    elif isinstance(metric, (list, np.ndarray)):
        res = np.array(metric)
        return res, True, "data"

    # 4. fallback safety net
    raise ValueError(
        f"Unrecognized data type for node metric: {type(metric).__name__}. "
        f"expected int, float, str ('strength'), dict, list, array, or pandas object."
    )


def _build_edges(mat, actual_thresh, directed, centroids, region_names, edge_curve, edge_thickness, edge_scaling):
    """constructs the 3d polydata tubes representing the connectome edges."""
    n_nodes = len(region_names)

    # identify indices of surviving edges based on the absolute threshold
    with np.errstate(invalid='ignore'):
        if directed:
            row_idx, col_idx = np.where((np.abs(mat) > actual_thresh) & (~np.eye(n_nodes, dtype=bool)))
        else:
            row_idx, col_idx = np.where(np.triu(np.abs(mat) > actual_thresh, k=1))

    if len(row_idx) == 0:
        return None, None, None

    surviving_weights = [abs(mat[i, j]) for i, j in zip(row_idx, col_idx)]
    if not surviving_weights:
        return None, None, None

    w_min, w_max = min(surviving_weights), max(surviving_weights)

    all_points, all_lines, all_scalars, all_radii = [], [], [], []
    pt_offset = 0

    # if edge_curve is 0, use 2 points (straight line). otherwise, use 10 points (smooth bezier curve)
    t = np.linspace(0, 1, 10 if edge_curve != 0.0 else 2)[:, None]

    for i, j in zip(row_idx, col_idx):
        val = mat[i, j]
        p1, p2 = centroids[region_names[i]], centroids[region_names[j]]

        # safely skip drawing edges connected to missing/nan regions
        if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
            continue

        # calculate 3d geometry of the edge
        if edge_curve != 0.0:
            # quadratic bezier curve math
            mid, vec = (p1 + p2) / 2.0, p2 - p1
            dist = np.linalg.norm(vec)
            if directed:
                # push curve outwards along the Z-axis cross product
                cross = np.cross(vec, np.array([0, 0, 1]))
                arc_mid = mid + (cross / np.linalg.norm(cross)) * (dist * edge_curve) if np.linalg.norm(cross) > 0 else mid
            else:
                # push curve inwards towards the origin
                norm_mid = np.linalg.norm(mid)
                arc_mid = mid - (mid / norm_mid) * (dist * edge_curve) if norm_mid > 0 else mid
            curve_pts = (1-t)**2 * p1 + 2*(1-t)*t * arc_mid + t**2 * p2
        else:
            # simple straight line between two points
            curve_pts = np.vstack((p1, p2))

        n_pts = len(curve_pts)
        all_points.append(curve_pts)
        all_lines.append([n_pts] + list(range(pt_offset, pt_offset + n_pts)))
        all_scalars.extend([val] * n_pts)

        # calculate physical radius of the tube
        if edge_thickness == 'weight':
            # normalize thickness between 0.1 and 0.9 based on connection strength
            norm_w = 0.1 + 0.8 * ((abs(val) - w_min) / (w_max - w_min)) if w_max > w_min else 1.0
            all_radii.extend([norm_w * edge_scaling] * n_pts)
        else:
            # constant user-provided thickness
            all_radii.extend([float(edge_thickness) * edge_scaling] * n_pts)

        pt_offset += n_pts

    if not all_points:
        return None, None, None

    # combine all individual edge segments into a single pyvista mesh
    edges_poly = pv.PolyData(np.vstack(all_points), lines=np.concatenate(all_lines))
    edges_poly.point_data['Connectivity'] = all_scalars
    edges_poly.point_data['TubeRadius'] = all_radii

    # extrude the lines into 3d tubes using the calculated radii
    return edges_poly.tube(scalars='TubeRadius', absolute=True), np.nanmin(all_scalars), np.nanmax(all_scalars)


def plot_connectome(matrix=None, atlas=None, custom_atlas_path=None, ax=None, cbar_kwargs=None, views=None, layout=None,
                    figsize=None, node_color='strength', node_size='strength', node_cmap='binary',
                    node_vminmax=[None, None], edge_threshold='95%', edge_thickness='weight',
                    edge_scaling=1.0, edge_cmap='coolwarm', edge_color=None, edge_alpha=1.0,
                    edge_vminmax=[None, None], edge_curve=0.1, directed=False,
                    style='default', bmesh_type='midthickness', bmesh_alpha=0.2,
                    bmesh_color='lightgray', zoom=1.2, display_type='matplotlib', export_path=None):
    """
    visualizes an n x n connectivity matrix as a 3d network in the brain.

    calculates spatial centroids for atlas regions and renders connectivity weights
    as 3d tubes (edges) and spheres (nodes) within a transparent brain hull.

    parameters
    ----------
    matrix : numpy.ndarray or pandas.dataframe, optional
        the (n, n) connectivity matrix. if none, only nodes are plotted.
        nan values are handled gracefully.
    atlas : str, optional
        name of the atlas mapping the regions (e.g., 'aparc', 'aseg').
    custom_atlas_path : str, optional
        path to custom atlas files if bypassing the built-in registry.
    views : list of str, optional
        list of views to render (e.g., ['left_lateral', 'superior']).
    layout : tuple, optional
        plotter grid layout (nrows, ncols). auto-generated if none.
    figsize : tuple, optional
        window size in inches (width, height). If None, automatically calculated based on layout.
    node_color : str, array, dict, optional
        can be 'atlas' (default categorical colors), 'strength' (graph metric), a static
        color string ('red'), or a custom data array/dict of matching length.
        default is 'strength' (when no matrix is provided, then 'atlas' is used).
    node_size : float, str, array, dict, optional
        constant float radius, 'strength' (graph metric),
        or a custom data array/dict to scale node sizes.
        default is 'strength' (when no matrix is provided, then 'atlas' is used).
    node_cmap : str, optional
        colormap name for mapped node colors. default is 'binary'.
    node_vminmax : list, optional
        [vmin, vmax] for node colormap clipping.
    edge_threshold : float or str, optional
        minimum absolute weight to display an edge. strings like '90%'
        calculate percentiles of the matrix. default is '95%'.
    edge_thickness : float or str, optional
        'weight' to scale by connection strength, or a constant float.
    edge_scaling : float, optional
        global multiplier for edge tube thickness. default is 1.0.
    edge_cmap : str, optional
        colormap name for edges. default is 'coolwarm'.
    edge_color : str, optional
        constant color for all edges, overriding the colormap.
    edge_alpha : float, optional
        opacity of the edges (0.0 to 1.0). default is 1.0.
    edge_vminmax : list, optional
        [vmin, vmax] for edge colormap clipping.
    edge_curve : float, optional
        amount of bend applied to edges. 0.0 draws straight lines. default is 0.1.
    directed : bool, optional
        if true, renders asymmetrical connections (full matrix instead of upper triangle).
    style : str, optional
        lighting/shading preset ('default', 'matte', 'glossy', etc.).
    bmesh_type : str, optional
        surface to render as context (e.g., 'midthickness'). default is 'midthickness'.
    bmesh_alpha : float, optional
        opacity of the context brain hull. default is 0.2.
    bmesh_color : str, optional
        color of the context brain hull. default is 'lightgray'.
    zoom : float, optional
        camera zoom level. default is 1.2.
    display_type : {'matplotlib', 'interactive', 'pyvista', 'object'}, optional
        'matplotlib': returns a matplotlib figure and axis (default).
        'interactive': opens an interactive trame viewer in the browser.
        'pyvista': returns a static jupyter widget (legacy behavior).
        'object': returns the raw pyvista plotter object.
    export_path : str, optional
        path to save the exported image.

    returns
    -------
    matplotlib.axes.Axes or pyvista.Plotter or IPython.display.DisplayObject
        returns based on display_type:
        - 'matplotlib': returns a matplotlib axes object.
        - 'interactive': returns a trame browser viewer.
        - 'pyvista': returns a static jupyter widget.
        - 'object': returns the raw pyvista plotter.
    """

    # detect atlas category and validate inputs
    bmesh_type = bmesh_type or 'midthickness'
    category = None
    if custom_atlas_path:
        files = os.listdir(custom_atlas_path)
        if any(f.endswith('.csv') for f in files): category = 'cortical'
        elif any(f.endswith('.vtk') or f.endswith('.gii') for f in files): category = 'subcortical'
        else: raise ValueError("could not detect atlas type in custom path.")
    else:
        atlas = atlas or 'aparc'
        resources = get_available_resources()
        if atlas in resources.get('cortical', []): category = 'cortical'
        elif atlas in resources.get('subcortical', []): category = 'subcortical'
        else: raise ValueError(f"atlas '{atlas}' not found in registry.")

    # load visual context brain securely
    bmesh = {}
    if bmesh_type:
        b_lh_path, b_rh_path = get_surface_paths(bmesh_type, 'bmesh')
        bmesh['L'] = load_gii2pv(b_lh_path)
        bmesh['R'] = load_gii2pv(b_rh_path)

    # compute spatial centers
    centroids, region_names, atlas_colors = _extract_centroids(category, atlas, custom_atlas_path, bmesh_type)

    # matrix parsing and nan-proof thresholding
    n_nodes = len(region_names)
    if matrix is not None:
        if isinstance(matrix, pd.DataFrame):
            mat = matrix.reindex(index=region_names, columns=region_names).values if set(region_names).intersection(matrix.index) else matrix.values
        else:
            mat = np.array(matrix, dtype=float)

        if mat.shape != (n_nodes, n_nodes):
            raise ValueError(f"matrix shape {mat.shape} does not match atlas regions ({n_nodes}).")

        if isinstance(edge_threshold, str) and edge_threshold.endswith('%'):
            perc = float(edge_threshold.strip('%'))
            upper_tri = np.abs(mat[np.triu_indices_from(mat, k=1)])
            valid_edges = upper_tri[~np.isnan(upper_tri)]
            actual_thresh = np.percentile(valid_edges, perc) if len(valid_edges) > 0 else 0
        else:
            actual_thresh = float(edge_threshold)
    else:
        mat, actual_thresh = np.zeros((n_nodes, n_nodes)), 1.0

    # build node geometry
    node_cloud = pv.PolyData(np.array([centroids[n] for n in region_names]))

    # parse node sizes (constant, 'strength', or custom data array)
    raw_sizes, is_size_mapped, size_name = _parse_node_metrics(node_size, mat, actual_thresh, directed, n_nodes, region_names)
    if is_size_mapped:
        s_min, s_max = np.nanmin(raw_sizes), np.nanmax(raw_sizes)
        # scale radii between 0 and 4 based on the data
        node_cloud.point_data['Radius'] = 0.0 + 4.0 * (raw_sizes - s_min) / (s_max - s_min) if s_max > s_min else np.full(n_nodes, 2.0)
    else:
        node_cloud.point_data['Radius'] = raw_sizes

    # parse node colors
    is_node_mapped, color_name = False, None
    n_vmin, n_vmax = None, None

    if matrix is None:
        node_color = 'atlas'
        node_size = 2.0

    if isinstance(node_color, str) and node_color == 'atlas':
        # use default atlas categorical colors
        node_cloud.point_data['Color'] = np.array(
            [np.array(to_rgba(atlas_colors.get(n, '#cccccc'))[:3]) * 255 for n in region_names]
        ).astype(np.uint8)
        rgb_mode = True

    elif isinstance(node_color, str) and node_color != 'strength':
        # use a constant user-provided color string (e.g., 'red')
        node_cloud.point_data['Color'] = np.array(
            [np.array(to_rgba(node_color)[:3]) * 255 for _ in region_names]
        ).astype(np.uint8)
        rgb_mode = True

    elif isinstance(node_color, dict) and node_color and isinstance(next(iter(node_color.values())), str):
        # use a custom dictionary mapping regions to specific color strings
        node_cloud.point_data['Color'] = np.array(
            [np.array(to_rgba(node_color.get(n, 'white'))[:3]) * 255 for n in region_names]
        ).astype(np.uint8)
        rgb_mode, is_node_mapped = True, False

    else:
        # map scalar values (like 'strength' or custom data) to a colormap
        raw_colors, is_color_mapped, color_name = _parse_node_metrics(node_color, mat, actual_thresh, directed, n_nodes, region_names)
        node_cloud.point_data['Color'] = raw_colors
        rgb_mode, is_node_mapped = False, True

        n_vmin = node_vminmax[0] if node_vminmax[0] is not None else np.nanmin(raw_colors)
        n_vmax = node_vminmax[1] if node_vminmax[1] is not None else np.nanmax(raw_colors)
        if n_vmin == n_vmax: n_vmin, n_vmax = n_vmin - 0.1, n_vmax + 0.1

    nodes_mesh = node_cloud.glyph(
        scale='Radius', geom=pv.Sphere(radius=1.0, theta_resolution=16, phi_resolution=16), orient=False
    )

    # build edge geometry
    merged_edges, e_vmin, e_vmax = None, None, None
    if matrix is not None:
        merged_edges, e_vmin, e_vmax = _build_edges(
            mat, actual_thresh, directed, centroids, region_names,
            edge_curve, edge_thickness, edge_scaling
        )
        if e_vmin is not None:
            e_vmin = edge_vminmax[0] if edge_vminmax[0] is not None else e_vmin
            e_vmax = edge_vminmax[1] if edge_vminmax[1] is not None else e_vmax
            if e_vmin == e_vmax: e_vmin, e_vmax = e_vmin - 0.1, e_vmax + 0.1

    # scene layout configuration
    sel_views = get_view_configs(views)
    ax, display_type, figsize = prepare_plotter(ax, display_type, sel_views, layout, figsize)

    is_edge_mapped = (merged_edges is not None) and (edge_color is None)
    edge_metric_name = "data" if edge_thickness == 'weight' else None
    needs_bottom = is_node_mapped or is_edge_mapped or size_name or edge_metric_name

    plotter, ncols, nrows = setup_plotter(sel_views, layout, figsize, display_type, needs_bottom_row=needs_bottom)
    shading_params = get_shading_preset(style)
    node_mapper, edge_mapper = None, None

    # render loop over views
    for i, (view_name, cfg) in enumerate(sel_views.items()):
        plotter.subplot(i // ncols, i % ncols)
        add_context_to_view(plotter, bmesh, cfg['side'], bmesh_alpha, bmesh_color, **shading_params)

        # render nodes
        node_props = shading_params.copy()
        node_props.update({'scalars': 'Color', 'rgb': rgb_mode, 'show_scalar_bar': False})
        if not rgb_mode: node_props.update({'cmap': node_cmap, 'clim': [n_vmin, n_vmax]})

        n_actor = plotter.add_mesh(nodes_mesh, **node_props)
        if not rgb_mode and node_mapper is None: node_mapper = n_actor.mapper

        # render edges
        if merged_edges is not None:
            edge_props = shading_params.copy()
            edge_props.update({'opacity': edge_alpha, 'show_scalar_bar': False})
            if edge_color is not None:
                edge_props.update({'color': edge_color})
            else:
                edge_props.update({'scalars': 'Connectivity', 'cmap': edge_cmap, 'clim': [e_vmin, e_vmax]})

            e_actor = plotter.add_mesh(merged_edges, **edge_props)
            if edge_color is None and edge_mapper is None: edge_mapper = e_actor.mapper

        set_camera(plotter, cfg, zoom=zoom)
        plotter.hide_axes()

    # colorbars
    cbar_info = []
    if needs_bottom:
        # use concise titles for colorbars to prevent layout squeezing
        edge_title = "edge weights" if is_edge_mapped else "edges"
        node_title = f"node {color_name}" if is_node_mapped and color_name else "nodes"

        if display_type != 'matplotlib':
            add_colorbars(plotter=plotter, mappers=[edge_mapper, node_mapper],
                          titles=[edge_title, node_title], nrows=nrows, figsize=figsize)
        else:
            if edge_mapper is not None:
                cbar_info.append({'cmap': edge_cmap, 'vminmax': [e_vmin, e_vmax], 'title': edge_title})
            if node_mapper is not None:
                cbar_info.append({'cmap': node_cmap, 'vminmax': [n_vmin, n_vmax], 'title': node_title})

    return finalize_plot(plotter, export_path=export_path, display_type=display_type, ax=ax, cbar_info=cbar_info, cbar_kwargs=cbar_kwargs)
