import os
import gc
import numpy as np
import pandas as pd
import nibabel as nib
import pyvista as pv
from matplotlib.colors import ListedColormap

from .data import (
    _resolve_resource_path, _find_cortical_files, 
    _find_subcortical_files, _find_tract_files
)

from .utils import (
    load_gii, load_gii2pv, prep_data, 
    generate_distinct_colors, parse_lut, map_values_to_surface, 
    lines_from_streamlines, make_cortical_mesh
)

from .scene import (
    get_view_configs, setup_plotter, add_context_to_view, 
    set_camera, finalize_plot, get_shading_preset
)


# --- plot for cortical surface ---

def plot_cortical(data=None, atlas=None, custom_atlas_path=None, views=None, layout=None, 
                  figsize=(1000, 600), cmap='RdYlBu_r', vminmax=[None, None], 
                  nan_color=(1.0, 1.0, 1.0), style='default', zoom=1.2, 
                  display_type='static', export_path=None):
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
    figsize : tuple (width, height), optional
        Window size in pixels. Default is (1000, 600).
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
    display_type : {'static', 'interactive', 'none'}, optional
        'static': Returns a static image (good for notebooks).
        'interactive': Opens an interactive viewer.
        'none': Renders off-screen (useful for batch export).
    export_path : str, optional
        If provided, saves the final figure to this path (e.g., 'figure.png').

    Returns
    -------
    pyvista.Plotter
        The plotter instance used for rendering.
    """

    # defaults
    if atlas is None and custom_atlas_path is None:
        atlas = 'aparc'

    # load brain mesh
    bmesh_path = _resolve_resource_path('conte69', 'bmesh')
    lh_v, lh_f = load_gii(os.path.join(bmesh_path, 'conte69.lh.gii'))
    rh_v, rh_f = load_gii(os.path.join(bmesh_path, 'conte69.rh.gii'))

    # resolve atlas path (either download or custom directory)
    atlas_dir = _resolve_resource_path(atlas, 'cortical', custom_path=custom_atlas_path)
    
    # locate files
    check_name = None if custom_atlas_path else atlas
    csv_path, lut_path = _find_cortical_files(atlas_dir, strict_name=check_name)

    # load mapping data
    tar_labels = np.loadtxt(csv_path, dtype=int)
    lut_ids, lut_colors, lut_names, max_id = parse_lut(lut_path)

    # map data
    all_vals = map_values_to_surface(data, tar_labels, lut_ids, lut_names)
    lh_vals = all_vals[:len(lh_v)]
    rh_vals = all_vals[len(lh_v):]

    # setup colors
    is_categorical = (data is None)
    n_colors = 256
    if is_categorical:
        _lut_colors = lut_colors.copy()
        _lut_colors[0] = nan_color
        cmap = ListedColormap(_lut_colors)
        n_colors = len(_lut_colors)
        vmin, vmax = 0, max_id
    else:
        if cmap is None: cmap = 'RdYlBu_r'
        vmin = vminmax[0] if vminmax[0] is not None else np.nanmin(all_vals)
        vmax = vminmax[1] if vminmax[1] is not None else np.nanmax(all_vals)

    # create meshes
    lh_mesh = make_cortical_mesh(lh_v, lh_f, lh_vals)
    rh_mesh = make_cortical_mesh(rh_v, rh_f, rh_vals)

    # setup plotter
    sel_views = get_view_configs(views)
    plotter, ncols, nrows = setup_plotter(sel_views, layout, figsize, display_type)
    shading_params = get_shading_preset(style)
    scalar_bar_mapper = None

    for i, (name, cfg) in enumerate(sel_views.items()):
        plotter.subplot(i // ncols, i % ncols)
        
        meshes = []
        if cfg['side'] in ['L', 'both']: meshes.append(lh_mesh)
        if cfg['side'] in ['R', 'both']: meshes.append(rh_mesh)

        for mesh in meshes:     
            actor = plotter.add_mesh(
                mesh,
                scalars='Data', 
                cmap=cmap, 
                clim=(vmin, vmax), 
                n_colors=n_colors,
                nan_color=nan_color, 
                show_scalar_bar=False,
                rgb=False, 
                smooth_shading=True,
                show_edges=False,
                interpolate_before_map=False,
                **shading_params
            )
            if scalar_bar_mapper is None: scalar_bar_mapper = actor.mapper

        set_camera(plotter, cfg, zoom=zoom)
        plotter.hide_axes()

    if not is_categorical and scalar_bar_mapper:
        plotter.subplot(nrows - 1, 0)
        plotter.add_scalar_bar(mapper=scalar_bar_mapper, title='', n_labels=2,
                               vertical=False, position_x=0.3, position_y=0.25, 
                               height=0.5, width=0.4,color='black', 
                               label_font_size=20)
    
    return finalize_plot(plotter, export_path, display_type)



# --- plot for subcortical structures ---

def plot_subcortical(data=None, atlas=None, custom_atlas_path=None, views=None, layout=None, 
                     figsize=(1000, 600), cmap='coolwarm', vminmax=[None, None], nan_color='#cccccc', 
                     nan_alpha=1.0, legend=False, style='default', bmesh_type='conte69', 
                     bmesh_alpha=0.1, bmesh_color='lightgray', zoom=1.2, display_type='static', 
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
        Window size in pixels. Default is (1000, 600).
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap for continuous data. Ignored if `data` is None. Default is 'coolwarm'.
    vminmax : list [min, max], optional
        Manual lower and upper bounds for the colormap. If [None, None], 
        bounds are inferred from the data range.
    nan_color : str or tuple, optional
        Color for regions with no data (NaN). Default is light grey '#cccccc'.
    nan_alpha : float, optional
        Opacity (0.0 to 1.0) for regions with no data. Set to 0.0 to hide them.
    legend : bool, optional
        If True (and data is None), displays a legend of region names and colors.
    style : str, optional
        Lighting preset ('default', 'matte', 'glossy', 'sculpted', 'flat').
    bmesh_type : str or None, optional
        Name of the background context brain mesh (e.g., 'conte69'). 
        Set to None to hide the context brain.
    bmesh_alpha : float, optional
        Opacity of the context brain mesh. Default is 0.1.
    bmesh_color : str, optional
        Color of the context brain mesh.
    zoom : float, optional
        Camera zoom level. >1.0 zooms in, <1.0 zooms out. Default is 1.2.
    display_type : {'static', 'interactive', 'none'}, optional
        'static': Returns a static image (good for notebooks).
        'interactive': Opens an interactive viewer.
        'none': Renders off-screen (useful for batch export).
    export_path : str, optional
        If provided, saves the final figure to this path (e.g., 'figure.png').
    custom_atlas_proc : dict, optional
        Parameters for processing custom GIfTI files. 
        Keys: 'smooth_i' (iterations) and 'smooth_f' (relaxation factor).
        Default is {'smooth_i': 15, 'smooth_f': 0.6}.

    Returns
    -------
    pyvista.Plotter
        The active plotter instance.
    """
    
    # defaults
    if atlas is None and custom_atlas_path is None:
        atlas = 'aseg'

    # load context brain mesh (if requested)
    bmesh = {}
    if bmesh_type:
        bmesh_path = _resolve_resource_path(bmesh_type, 'bmesh')
        for h in ['lh', 'rh']:
            fpath = os.path.join(bmesh_path, f'{bmesh_type}.{h}.gii')
            if os.path.exists(fpath):
                bmesh[h] = load_gii2pv(fpath)
    
    # load regional atlas meshes

    # resolve atlas path (either download or custom directory)
    atlas_dir = _resolve_resource_path(atlas, 'subcortical', custom_path=custom_atlas_path)

    # locate mesh files, returns dict: {'Left_Thalamus': '/path/to/Left_Thalamus.vtk', ...}
    file_map = _find_subcortical_files(atlas_dir)
    rmesh_names = sorted(list(file_map.keys()))

    # load meshes (and convert gii2pv if gii files)
    meshes = {}
    for name, fpath in file_map.items():
        if fpath.endswith('.vtk'):
            meshes[name] = pv.read(fpath)
        elif fpath.endswith('.gii'):
            mesh = load_gii2pv(fpath, **custom_atlas_proc)
            meshes[name] = mesh

    # prepare colors / map data
    if data is not None:
        d_data = prep_data(data, rmesh_names, atlas, 'subcortical')

        valid_vals = [v for v in d_data.values() if pd.notna(v)]
        if vminmax[0] is None: vminmax[0] = min(valid_vals) if valid_vals else 0
        if vminmax[1] is None: vminmax[1] = max(valid_vals) if valid_vals else 1
    else:
        colors = generate_distinct_colors(len(rmesh_names), seed=42)
        d_atlas_colors = {name: color for name, color in zip(rmesh_names, colors)}

    # setup plotter
    sel_views = get_view_configs(views)
    needs_bottom = (data is not None) or legend
    plotter, ncols, nrows = setup_plotter(sel_views, layout, figsize, display_type, 
                                           needs_bottom_row=needs_bottom)
    
    # get shading parameters from style
    shading_params = get_shading_preset(style)
    
    scalar_bar_mapper = None
    plotted_regions = {}

    # plotting loop
    for i, (view_name, cfg) in enumerate(sel_views.items()):
        plotter.subplot(i // ncols, i % ncols)

        # add context (uses style kwargs for consistent lighting)
        add_context_to_view(plotter, bmesh, cfg['side'], bmesh_alpha, bmesh_color, 
                            **shading_params)

        # add regions
        for name, mesh in meshes.items():
            # side filter
            # TODO: make the hemisphere specific name check more robust
            name_lower = name.lower()
            is_left = any(x in name_lower for x in ['left', '_l', '-l', 'l_']) or name_lower.endswith('l')
            is_right = any(x in name_lower for x in ['right', '_r', '-r', 'r_']) or name_lower.endswith('r')
            
            if cfg['side'] == 'L' and is_right and not is_left: continue
            if cfg['side'] == 'R' and is_left and not is_right: continue

            # determine properties for this mesh
            props = shading_params.copy()
            
            if data is not None:
                val = d_data.get(name, np.nan) if pd.notna(d_data.get(name)) else np.nan
                has_val = not np.isnan(val)
                
                mesh['Data'] = np.full(mesh.n_points, val)
                
                props.update({
                    'scalars': 'Data', 'cmap': cmap, 'clim': vminmax,
                    'nan_color': nan_color, 'opacity': 1.0 if has_val else nan_alpha, 
                    'show_scalar_bar': False
                })
            else:
                color = d_atlas_colors[name]
                props.update({'color': color, 'opacity': 1.0})
                plotted_regions[name] = color

            actor = plotter.add_mesh(mesh, **props)
            
            if data is not None and scalar_bar_mapper is None and 'scalars' in props:
                 scalar_bar_mapper = actor.mapper

        set_camera(plotter, cfg, zoom=zoom)
        plotter.hide_axes()

    # bottom row: legend or colorbar
    if needs_bottom:
        plotter.subplot(nrows - 1, 0)
        
        if data is not None:
            if scalar_bar_mapper:
                plotter.add_scalar_bar(mapper=scalar_bar_mapper, title='', n_labels=5, 
                                       vertical=False, position_x=0.3, position_y=0.25, 
                                       height=0.5, width=0.4, color='black', 
                                       label_font_size=20)
        elif legend:
            legend_entries = [[r, c] for r, c in plotted_regions.items()]
            if legend_entries:
                plotter.add_legend(legend_entries, size=(0.8, 0.8), bcolor=None)

    return finalize_plot(plotter, export_path, display_type)



# --- plot for white matter tracts ---

_TRACT_CACHE = {}
def clear_tract_cache():
    """manually clears the global geometry cache to free ram."""
    global _TRACT_CACHE
    _TRACT_CACHE = {}
    gc.collect()
    print("Tract cache cleared.")


def plot_tracts(data=None, atlas=None, custom_atlas_path=None, views=None, layout=None, 
                figsize=(1000, 800), cmap='coolwarm', alpha=1.0, vminmax=[None, None], 
                nan_color='#BDBDBD', nan_alpha=1.0, legend=False, style='default',
                bmesh_type='conte69', bmesh_alpha=0.2, bmesh_color='lightgray', 
                zoom=1.2, orientation_coloring=False, display_type='static', 
                tract_kwargs=dict(render_lines_as_tubes=True, line_width=1.2),
                export_path=None):
    """
    Visualize data on the white matter tractography bundles using a specified atlas.

    Renders streamlines from .trk files. Can color tracts by scalar values, 
    categorically, or by local fiber orientation.

    Parameters
    ----------
    data : dict, list, numpy.ndarray, pandas.Series, pandas.DataFrame, optional
        Scalar values for each tract.
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
        Window size in pixels. Default is (1000, 600).
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
    legend : bool, optional
        If True (and data is None), displays a legend of region names and colors.
    style : str, optional
        Lighting preset ('default', 'matte', 'glossy', 'sculpted', 'flat').
    bmesh_type : str or None, optional
        Name of the background context brain mesh (e.g., 'conte69'). 
        Set to None to hide the context brain.
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
    display_type : {'static', 'interactive', 'none'}, optional
        'static': Returns a static image (good for notebooks).
        'interactive': Opens an interactive viewer.
        'none': Renders off-screen (useful for batch export).
    export_path : str, optional
        If provided, saves the final figure to this path (e.g., 'figure.png').

    Returns
    -------
    pyvista.Plotter
        The active plotter instance.
    """
    
    # defaults
    if atlas is None and custom_atlas_path is None:
        atlas = 'xtract_tiny'

    # resolve atlas path (either download or custom directory)
    atlas_dir = _resolve_resource_path(atlas, 'tracts', custom_path=custom_atlas_path)

    # locate tract files, returns dict eg {'CST_L': '/path/to/CST_L.trk', ...}
    file_map = _find_tract_files(atlas_dir)
    tract_names = sorted(list(file_map.keys()))

    # prepare colors / map data
    if data is not None:
        d_data = prep_data(data, tract_names, atlas, 'tracts')

        valid_vals = [v for v in d_data.values() if pd.notna(v)]
        vmin = vminmax[0] if vminmax[0] is not None else min(valid_vals)
        vmax = vminmax[1] if vminmax[1] is not None else max(valid_vals)
    # categorical/orientation mode
    else:
        vmin, vmax = 0, 1 
        colors = generate_distinct_colors(len(tract_names), seed=42)
        d_atlas_colors = {name: color for name, color in zip(tract_names, colors)}

    # load context brain mesh (if requested)
    bmesh = {}
    if bmesh_type:
        bmesh_path = _resolve_resource_path(bmesh_type, 'bmesh')
        for h in ['lh', 'rh']:
            fpath = os.path.join(bmesh_path, f'{bmesh_type}.{h}.gii')
            if os.path.exists(fpath):
                bmesh[h] = load_gii2pv(fpath)

    # setup plotter
    sel_views = get_view_configs(views)
    needs_bottom = (data is not None and not orientation_coloring) or legend
    plotter, ncols, nrows = setup_plotter(sel_views, layout, figsize, display_type, 
                                           needs_bottom_row=needs_bottom)
    plotter.enable_depth_peeling(number_of_peels=10)
    plotter.enable_anti_aliasing('msaa') # smooth lines
    shading_params = get_shading_preset(style)

    scalar_bar_mapper = None
    plotted_regions = {} # for legend

    def _retrieve_tract_mesh(atlas_key, name, file_map):
        """
        Retrieves a mesh from cache or loads from disk using file_map.
        """
        # check RAM cache
        if name in _TRACT_CACHE.get(atlas_key, {}):
            return _TRACT_CACHE[atlas_key][name]

        # init cache dict
        if atlas_key not in _TRACT_CACHE: _TRACT_CACHE[atlas_key] = {}

        # load from disk
        try:
            fpath = file_map.get(name)
            if not fpath: return None

            tractogram = nib.streamlines.load(fpath)
            points, lines, tangents = lines_from_streamlines(tractogram.streamlines)
            if len(points) == 0: return None
            
            base_mesh = pv.PolyData(points, lines=lines)
            base_mesh.point_data['tangents'] = np.abs(tangents)
            
            # store in global cache
            _TRACT_CACHE[atlas_key][name] = base_mesh
            return base_mesh
            
        except Exception as e:
            print(f"Failed to load tract {name}: {e}")
            return None

    # plotting loop
    cache_key = 'custom' if custom_atlas_path else atlas
    for i, (view_name, cfg) in enumerate(sel_views.items()):
        plotter.subplot(i // ncols, i % ncols)
        
        # add context (passed shading params to context mesh)
        add_context_to_view(plotter, bmesh, cfg['side'], bmesh_alpha, bmesh_color, **shading_params)

        # add tracts
        for name in tract_names:
            # optimization: early exit for hidden tracts
            has_value = False
            val = np.nan
            
            if data is not None and not orientation_coloring:
                if name in d_data and pd.notna(d_data[name]):
                    val = d_data[name]
                    has_value = True
                elif nan_alpha == 0:
                    continue 
            
            # side filtering
            name_lower = name.lower()
            is_left = any(x in name_lower for x in ['left', '_l', '-l', 'l_']) or name_lower.endswith('l')
            is_right = any(x in name_lower for x in ['right', '_r', '-r', 'r_']) or name_lower.endswith('r')
            if cfg['side'] == 'L' and is_right and not is_left: continue
            if cfg['side'] == 'R' and is_left and not is_right: continue

            # load mesh
            base_mesh = _retrieve_tract_mesh(cache_key, name, file_map)
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
                legend_color = 'gray'

            elif data is not None:
                pv_mesh['Data'] = np.full(pv_mesh.n_points, val)
                current_opacity = alpha if has_value else nan_alpha
                
                props.update({
                    'scalars': 'Data', 'cmap': cmap, 'clim': (vmin, vmax),
                    'nan_color': nan_color, 'opacity': current_opacity, 'show_scalar_bar': False
                })
                legend_color = None

            else:
                color = d_atlas_colors[name]
                props.update({
                    'color': color, 'opacity': alpha
                })
                legend_color = color

            actor = plotter.add_mesh(pv_mesh, **props)
            
            if legend_color: plotted_regions[name] = legend_color
            if data is not None and not orientation_coloring and scalar_bar_mapper is None and 'scalars' in props:
                scalar_bar_mapper = actor.mapper

        set_camera(plotter, cfg, zoom=zoom, distance=150)
        plotter.hide_axes()

    # bottom row: legend or colorbar
    if needs_bottom:
        plotter.subplot(nrows - 1, 0)
        
        if data is not None and not orientation_coloring:
            if scalar_bar_mapper:
                plotter.add_scalar_bar(mapper=scalar_bar_mapper, title='', n_labels=5, vertical=False,
                                       position_x=0.3, position_y=0.25, height=0.5, width=0.4,
                                       color='black', label_font_size=20)
        elif legend and not orientation_coloring:
            legend_entries = [[r, c] for r, c in plotted_regions.items()]
            if legend_entries:
                plotter.add_legend(legend_entries, size=(0.8, 0.8), bcolor=None)

    # finalize
    ret_val = finalize_plot(plotter, export_path, display_type)
    
    if display_type != 'interactive':
        del plotter
        gc.collect()

    return ret_val