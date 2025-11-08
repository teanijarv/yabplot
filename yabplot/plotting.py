import os
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from yabplot.utils import get_resource_path, load_gii2pv, prep_data, generate_distinct_colors

def plot_roi(data=None, views=None, layout=None, figsize=(300, 300), cmap='coolwarm', 
             vminmax=[None, None], nan_color='#cccccc', legend=False, lighting=True,
             bmesh_type='conte69_32k', bmesh_alpha=0.1, bmesh_color='lightgray', 
             bmesh_kwargs=dict(specular=0.0, ambient=0.6, diffuse=0.6),
             rmesh_kwargs=dict(specular=0.0, ambient=0.8, diffuse=0.5),
             display_type='static', export_path=None, rseed=42):

    # brain mesh
    bmesh = {}
    if bmesh_type is not None:
        for h in ['lh', 'rh']:
            bmesh[h] = load_gii2pv(get_resource_path(f'brainmesh/{bmesh_type}.{h}.gii'))

    # regional meshes
    rmesh_dir = get_resource_path(f'musus100_dbn/surfs')
    rmesh_files = sorted([f for f in os.listdir(rmesh_dir) 
                        if f.endswith('_surface.surf.gii')])
    rmesh_names = [f.replace('_surface.surf.gii', '') for f in rmesh_files]

    # colors for data
    if data is not None:
        d_data = prep_data(data)
        colormap = plt.get_cmap(cmap)
        
        # get valid (non-nan) values and set minmax with color range
        valid_vals = [v for v in d_data.values() if pd.notna(v)]
        if vminmax[0] is None:
            vminmax[0] = min(valid_vals)# if valid_vals else 0
        if vminmax[1] is None:
            vminmax[1] = max(valid_vals)# if valid_vals else 1
        norm = mcolors.Normalize(vmin=vminmax[0], vmax=vminmax[1])
    
    # colors for atlas labels
    else:
        colors = generate_distinct_colors(len(rmesh_names), seed=rseed)
        d_atlas_colors = {name: color for name, color in zip(rmesh_names, colors)}

    # load regional meshes and determine colors
    mesh_data = {}
    plotted_regions = {}
    for rmesh_file in rmesh_files:
        name = rmesh_file.replace('_surface.surf.gii', '')

        # load mesh
        pv_mesh = load_gii2pv(os.path.join(rmesh_dir, rmesh_file), smooth=True)
        
        # color based on data value
        if data is not None:
            if name in data and pd.notna(data[name]):
                val = data[name]
                rgba = colormap(norm(val))
                color = rgba[:3] # only rgb
            else:
                color = nan_color  # Gray for NaN
        # distinct color for atlas
        else:
            color = d_atlas_colors[name]
        
        mesh_data[name] = (pv_mesh, color)
        plotted_regions[name] = color

    # views
    all_views = {
        'left_lateral': {'elev': 0, 'azim': 180, 'title': 'Left Lateral', 'side': 'L'},
        'left_medial': {'elev': 0, 'azim': 0, 'title': 'Left Medial', 'side': 'L'},
        'right_lateral': {'elev': 0, 'azim': 0, 'title': 'Right Lateral', 'side': 'R'},
        'right_medial': {'elev': 0, 'azim': 180, 'title': 'Right Medial', 'side': 'R'},
        'superior': {'elev': 90, 'azim': 0, 'title': 'Superior', 'side': 'both'},
        'inferior': {'elev': -90, 'azim': 0, 'title': 'Inferior', 'side': 'both'}
    }
    if views is None: views = list(all_views.keys())
    sel_views = {k: all_views[k] for k in views if k in all_views}

    # layout
    if layout is None:
        n_views = len(sel_views)
        if n_views <= 4: layout = (1, n_views)
        elif n_views <= 6: layout = (2, 3)
        else: layout = (int(np.ceil(n_views / 4)), 4)
    if legend and data is None: _layout = (layout[0], layout[1]+1)
    else: _layout = layout

    # plot setup
    plotter = pv.Plotter(
        shape=_layout, 
        off_screen=True,
        window_size=figsize,
        border=False
    )
    plotter.set_background('white')

    # add meshes to plots
    for i, (view_name, view) in enumerate(sel_views.items()):
        row = i // layout[1]
        col = i % layout[1]
        plotter.subplot(row, col)

        # brain mesh
        if bmesh_type is not None:
            for h, mesh in bmesh.items():
                mesh_args = dict(mesh=mesh, color=bmesh_color, 
                    opacity=bmesh_alpha, smooth_shading=True,
                    show_edges=False, lighting=lighting, **bmesh_kwargs)
                if view['side'] == 'L' and h == 'lh':
                    plotter.add_mesh(**mesh_args)
                elif view['side'] == 'R' and h == 'rh':
                    plotter.add_mesh(**mesh_args)
                elif view['side'] == 'both':
                    plotter.add_mesh(**mesh_args)
        
        # regional meshes
        for name, (pv_mesh, color) in mesh_data.items():
            l_prefixes = ('L-', 'L_', 'LH-', 'LH_', 'lh-', 'lh_', 'Left-', 'Left_')
            r_prefixes = ('R-', 'R_', 'RH-', 'RH_', 'rh-', 'rh_', 'Right-', 'Right_')
            if name.startswith(l_prefixes):
                hemi = 'L'
            elif name.startswith(r_prefixes):
                hemi = 'R'
            else:
                hemi = 'LR'
            
            if view['side'] == 'L' and 'L' not in hemi:
                continue
            if view['side'] == 'R' and 'R' not in hemi:
                continue

            if data is not None:
                rval = np.nan
                if name in data and pd.notna(data[name]):
                    rval = data[name]
                pv_mesh.point_data['data'] = np.ones(pv_mesh.n_points) * rval
                
                plotter.add_mesh(pv_mesh, scalars='data', cmap=cmap, clim=vminmax, 
                    nan_color=nan_color, show_scalar_bar=False, lighting=lighting, **rmesh_kwargs)
            else:
                plotter.add_mesh(pv_mesh, color=color, lighting=lighting, **rmesh_kwargs)
            
        # camera
        distance = 120
        if view_name == 'left_lateral':
            plotter.camera.position = (-distance, 0, 0)
            plotter.camera.focal_point = (0, 0, 0)
            plotter.camera.up = (0, 0, 1)
        elif view_name == 'left_medial':
            plotter.camera.position = (distance, 0, 0)
            plotter.camera.focal_point = (0, 0, 0)
            plotter.camera.up = (0, 0, 1)
        elif view_name == 'right_lateral':
            plotter.camera.position = (distance, 0, 0)
            plotter.camera.focal_point = (0, 0, 0)
            plotter.camera.up = (0, 0, 1)
        elif view_name == 'right_medial':
            plotter.camera.position = (-distance, 0, 0)
            plotter.camera.focal_point = (0, 0, 0)
            plotter.camera.up = (0, 0, 1)
        elif view_name == 'superior':
            plotter.camera.position = (0, 0, distance)
            plotter.camera.focal_point = (0, 0, 0)
            plotter.camera.up = (0, 1, 0)
        elif view_name == 'inferior':
            plotter.camera.position = (0, 0, -distance)
            plotter.camera.focal_point = (0, 0, 0)
            plotter.camera.up = (0, 1, 0)
        plotter.camera.parallel_projection = True
        
        if bmesh_type is not None:
            plotter.reset_camera()
        else:
            plotter.camera.parallel_scale = 40

    plotter.hide_axes()

    # add legend or colorbar
    if legend:
        if data is not None:
            plotter.add_scalar_bar(width=0.4, position_y=0.015, vertical=False)
        else:
            plotter.subplot(0, _layout[1]-1)
            legend_entries = []
            for region, color in plotted_regions.items():
                legend_entries.append([region, color])
            plotter.add_legend(legend_entries, size=(0.8, 0.8))

    # export and render
    if export_path is not None:
        plotter.screenshot(export_path, transparent_background=True)
    if display_type == 'static':
        plotter.show(jupyter_backend='static')
    # elif display_type == 'html':
    #     plotter.show(jupyter_backend='html')
    elif display_type == 'interactive':
        plotter.show(jupyter_backend='interactive', return_viewer=True)
        plotter.show()
    
    return plotter