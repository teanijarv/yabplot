import os
import numpy as np
import pyvista as pv
import warnings

def get_shading_preset(style_name):
    """
    Returns a dictionary of lighting parameters for pyvista.add_mesh.
    
    Styles:
    - 'default': Balanced, no shine.
    - 'matte':   (Soft) High ambient, low contrast. Good for reading atlas colors.
    - 'sculpted':(Hard) Stronger shadows, higher contrast. Good for showing anatomy.
    - 'glossy':  (Shiny) Wet/Plastic look with specular highlights. 
    """
    presets = {
        'default': {
            'lighting': True,
            'specular': 0.0, 
            'ambient': 0.65, 
            'diffuse': 0.4, 
            'specular_power': 15
        },
        # very bright shadows
        'matte': {
            'lighting': True,
            'specular': 0.0, 
            'ambient': 0.75,
            'diffuse': 0.2, 
            'specular_power': 0
        },
        # slight shime, dark shadows, strong directional light
        'sculpted': {
            'lighting': True,
            'specular': 0.05,
            'ambient': 0.4,
            'diffuse': 0.6,
            'specular_power': 10
        },
        # strong shine, sharp highlights
        'glossy': {
            'lighting': True,
            'specular': 0.3,
            'ambient': 0.4, 
            'diffuse': 0.6, 
            'specular_power': 30
        },
        # flat 2D
        'flat': {
            'lighting': False,
            'ambient': 1.0, 
            'diffuse': 0.0, 
            'specular': 0.0
        }
    }
    
    if style_name not in presets:
        print(f"Warning: Style '{style_name}' not found. Using 'default'. Options: {list(presets.keys())}")
        return presets['default']
        
    return presets[style_name]

def get_view_configs(view_names):
    all_views = {
        'left_lateral':  {'pos': (-1, 0, 0), 'up': (0, 0, 1), 'side': 'L'},
        'right_lateral': {'pos': (1, 0, 0),  'up': (0, 0, 1), 'side': 'R'},
        'left_medial':   {'pos': (1, 0, 0),  'up': (0, 0, 1), 'side': 'L'},
        'right_medial':  {'pos': (-1, 0, 0), 'up': (0, 0, 1), 'side': 'R'},
        'superior':      {'pos': (0, 0, 1),  'up': (0, 1, 0), 'side': 'both'},
        'inferior':      {'pos': (0, 0, -1), 'up': (0, 1, 0), 'side': 'both'},
        'anterior':      {'pos': (0, 1, 0),  'up': (0, 0, 1), 'side': 'both'},
        'posterior':     {'pos': (0, -1, 0), 'up': (0, 0, 1), 'side': 'both'}
    }
    if view_names is None: return all_views
    return {k: all_views[k] for k in view_names if k in all_views}

def setup_plotter(sel_views, layout, figsize, display_type, needs_bottom_row=True):
    n = len(sel_views)
    if layout is None:
        if n <= 4: base_layout = (1, n)
        elif n <= 6: base_layout = (2, 3)
        else: base_layout = (int(np.ceil(n/4)), 4)
    else: base_layout = layout

    if needs_bottom_row:
        nrows, ncols = base_layout[0] + 1, base_layout[1]
        groups = [(nrows - 1, slice(0, ncols))]
        row_weights = [1.0]*base_layout[0] + [0.2]
    else:
        nrows, ncols = base_layout[0], base_layout[1]
        groups = None
        row_weights = None

    plotter = pv.Plotter(shape=(nrows, ncols), groups=groups, row_weights=row_weights,
                         off_screen=(display_type=='object' or display_type=='none'),
                         window_size=figsize, border=False)
    plotter.set_background('white')
    return plotter, ncols, nrows
        
def add_context_to_view(plotter, bmesh, view_side, alpha, color, **kwargs):
    """
    Adds context mesh. Lighting parameters are passed via **kwargs.
    """
    if not bmesh: return
    for h, mesh in bmesh.items():
        if (view_side == 'L' and h == 'R') or (view_side == 'R' and h == 'L'): continue
        plotter.add_mesh(mesh, color=color, opacity=alpha, 
                         smooth_shading=True, show_edges=False, 
                         **kwargs)

def set_camera(plotter, view_cfg, zoom=1.0, distance=200):
    plotter.camera.position = tuple(p * distance for p in view_cfg['pos'])
    plotter.camera.focal_point = (0, 0, 0)
    plotter.camera.up = view_cfg['up']
    plotter.camera.parallel_projection = True
    plotter.reset_camera()
    plotter.camera.zoom(zoom)

def add_colorbars(plotter, mappers, titles, nrows, figsize):
    """
    Adds unified, cleanly formatted colorbars to the bottom row of the plot.
    """
    plotter.subplot(nrows - 1, 0) 
    
    valid_mappers = [m for m in mappers if m is not None]
    valid_titles = [t for m, t in zip(mappers, titles) if m is not None]
    num_bars = len(valid_mappers)
    
    if num_bars == 0: 
        return

    # calculate dynamic sizing based on window aspect ratio
    width_px, height_px = figsize
    aspect_ratio = width_px / height_px
    cb_width = 0.35 if aspect_ratio > 1.5 else 0.60
    pos_x = (1.0 - cb_width) / 2.0  
    cb_height = 0.2 
    
    if num_bars == 1: 
        positions_y = [0.15] 
        cb_height = 0.4
    elif num_bars == 2: 
        positions_y = [0.5, 0.01]
    else: 
        positions_y = np.linspace(0.35, 0.05, num_bars)
    
    for i, (mapper, title) in enumerate(zip(valid_mappers, valid_titles)):
        plotter.add_scalar_bar(
            mapper=mapper, title=title, vertical=False, 
            position_x=pos_x, position_y=positions_y[i], 
            height=cb_height, width=cb_width, color='black', 
            title_font_size=13, label_font_size=11, n_labels=3, fmt="%g"
        )

def finalize_plot(plotter, export_path, display_type):
    if export_path: 
        ext = os.path.splitext(export_path)[1].lower()
        vector_formats = ['.svg', '.eps', '.ps', '.pdf', '.tex']
        raster_formats = ['.png', '.jpeg', '.jpg', '.bmp', '.tiff']
        
        # save as vector graphic or rasterized image
        if ext in vector_formats:
            plotter.save_graphic(export_path)
        elif ext in raster_formats:
            plotter.screenshot(export_path, transparent_background=True)
        else:
            supported = vector_formats + raster_formats
            warnings.warn(
                f"[WARNING] unsupported export extension '{ext}'. file not saved. "
                f"supported formats are: {', '.join(supported)}"
            )
    
    if display_type == 'static': 
        out = plotter.show(jupyter_backend='static')
        plotter.close()
    elif display_type == 'interactive': 
        out = plotter.show(jupyter_backend='trame')
    elif display_type == 'object': 
        plotter.render()
        return plotter
    else:
        plotter.close()
        out = None
    
    return out
