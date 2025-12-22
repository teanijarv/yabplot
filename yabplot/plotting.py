import os
import gc
import numpy as np
import pandas as pd
import nibabel as nib
import pyvista as pv
from matplotlib.colors import ListedColormap

from .utils import (
    get_resource_path, get_atlas_names, load_gii, load_gii2pv, prep_data, 
    generate_distinct_colors, parse_lut, map_values_to_surface, 
    lines_from_streamlines, make_cortical_mesh
)

from .scene import (
    get_view_configs, setup_plotter, load_context_mesh, add_context_to_view, 
    set_camera, finalize_plot, get_shading_preset
)


# --- plot for cortical surface ---

def plot_cortical(data=None, atlas='aparc', views=None, layout=None, 
                  figsize=(1000, 600), cmap='RdYlBu_r', vminmax=[None, None], 
                  nan_color=(1.0, 1.0, 1.0), style='default', zoom=1.2, 
                  display_type='static', export_path=None):

    av_atlases = get_atlas_names(type='verts')
    if atlas not in av_atlases:
        raise RuntimeError(f"Atlas '{atlas}' not available: {av_atlases}")

    # load geometry
    bmesh_path = get_resource_path(os.path.join('brainmesh'))
    lh_v, lh_f = load_gii(os.path.join(bmesh_path, f'conte69_32k.lh.gii'))
    rh_v, rh_f = load_gii(os.path.join(bmesh_path, f'conte69_32k.rh.gii'))

    # load mapping
    atlas_path = get_resource_path(os.path.join('atlas', 'verts', atlas))
    csv_path = os.path.join(atlas_path, f'{atlas}_conte69.csv')
    tar_labels = np.loadtxt(csv_path, dtype=int)
    lut_path = os.path.join(atlas_path, f'{atlas}_LUT.txt')
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

def plot_subcortical(data=None, atlas='aseg', views=None, layout=None, 
                     figsize=(1000, 600), cmap='coolwarm', vminmax=[None, None], 
                     nan_color='#cccccc', nan_alpha=1.0, legend=False, 
                     style='default', bmesh_type='conte69_32k', bmesh_alpha=0.1, 
                     bmesh_color='lightgray', zoom=1.2, display_type='static', 
                     export_path=None):
    
    # check availability
    av_atlases = get_atlas_names(type='surfs')
    if atlas not in av_atlases:
        raise RuntimeError(f"Atlas '{atlas}' not available: {av_atlases}")

    # load context brain mesh (if requested)
    bmesh = load_context_mesh(bmesh_type) if bmesh_type else {}

    # load regional atlas meshes
    def _load_subcortical_meshes(atlas):
        """locates and loads regional meshes (vtk preferred, gii fallback)."""
        # 1. try loading cached .vtk files
        pv_dir = get_resource_path(os.path.join('atlas', 'surfs', atlas, 'pv'))
        if os.path.exists(pv_dir):
            files = sorted([f for f in os.listdir(pv_dir) if f.endswith('.vtk')])
            if files:
                meshes = {}
                for f in files:
                    name = f.replace('.vtk', '')
                    meshes[name] = pv.read(os.path.join(pv_dir, f))
                return meshes

        # 2. fallback to .gii and cache
        gii_dir = get_resource_path(os.path.join('atlas', 'surfs', atlas, 'gii'))
        if os.path.exists(gii_dir):
            files = sorted([f for f in os.listdir(gii_dir) if f.endswith('_surface.surf.gii')])
            meshes = {}
            os.makedirs(pv_dir, exist_ok=True)
            for f in files:
                name = f.replace('_surface.surf.gii', '')
                mesh = load_gii2pv(os.path.join(gii_dir, f), smooth_i=15, smooth_f=0.6)
                mesh.save(os.path.join(pv_dir, f"{name}.vtk"))
                meshes[name] = mesh
            return meshes

        raise RuntimeError(f"No meshes found for atlas '{atlas}'.")

    meshes = _load_subcortical_meshes(atlas)
    rmesh_names = list(meshes.keys())

    # prepare colors
    if data is not None:
        d_data = prep_data(data)
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
    
    # Get shading parameters from style
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


def plot_tracts(data=None, 
                atlas='hcp1065_small', 
                views=None, 
                layout=None, 
                figsize=(1000, 800), 
                cmap='coolwarm', 
                alpha=1.0,
                vminmax=[None, None], 
                nan_color='#BDBDBD', 
                nan_alpha=1.0, 
                legend=True, 
                style='default',
                bmesh_type='conte69_32k', 
                bmesh_alpha=0.2, 
                bmesh_color='lightgray', 
                zoom=1.2,
                orientation_coloring=False, 
                tract_kwargs=dict(render_lines_as_tubes=True, line_width=1.2),
                display_type='static', 
                export_path=None):
    
    # check availability
    tract_dir = get_resource_path(os.path.join('atlas', 'tracts', atlas))
    if not os.path.exists(tract_dir):
         raise RuntimeError(f"Tract atlas directory not found: {tract_dir}")

    # identify tracts
    tract_files = sorted([f for f in os.listdir(tract_dir) if f.endswith('.trk')])
    tract_names = [f.replace('.trk', '') for f in tract_files]
    
    # prepare colors
    # data mode
    if data is not None and not orientation_coloring:
        d_data = prep_data(data)
        valid_vals = [v for v in d_data.values() if pd.notna(v)]
        if not valid_vals: valid_vals = [0, 1]
        vmin = vminmax[0] if vminmax[0] is not None else min(valid_vals)
        vmax = vminmax[1] if vminmax[1] is not None else max(valid_vals)
    # categorical/orientation mode
    else:
        vmin, vmax = 0, 1 
        colors = generate_distinct_colors(len(tract_names), seed=42)
        d_atlas_colors = {name: color for name, color in zip(tract_names, colors)}

    # load context brain mesh
    bmesh = load_context_mesh(bmesh_type) if bmesh_type else {}

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

    def _retrieve_tract_mesh(atlas, name, tract_dir):
        """
        retrieves a mesh from cache, disk cache, or raw source.
        returns the base mesh (to be copied).
        """
        # 1. check ram cache
        if name in _TRACT_CACHE.get(atlas, {}):
            return _TRACT_CACHE[atlas][name]

        # ensure atlas dict exists
        if atlas not in _TRACT_CACHE: _TRACT_CACHE[atlas] = {}

        # 2. load from disk (try source .trk)
        try:
            fpath = os.path.join(tract_dir, f"{name}.trk")
            if not os.path.exists(fpath): return None

            tractogram = nib.streamlines.load(fpath)
            points, lines, tangents = lines_from_streamlines(tractogram.streamlines)
            if len(points) == 0: return None
            
            base_mesh = pv.PolyData(points, lines=lines)
            base_mesh.point_data['tangents'] = np.abs(tangents)
            
            # store in global cache
            _TRACT_CACHE[atlas][name] = base_mesh
            return base_mesh
            
        except Exception as e:
            print(f"Failed to load tract {name}: {e}")
            return None

    # plotting loop
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
            base_mesh = _retrieve_tract_mesh(atlas, name, tract_dir)
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