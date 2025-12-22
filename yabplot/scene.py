import os
import numpy as np
import pyvista as pv
from yabplot.utils import get_resource_path, load_gii2pv

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
                         off_screen=(display_type=='none'), window_size=figsize, border=False)
    plotter.set_background('white')
    return plotter, ncols, nrows

def load_context_mesh(bmesh_type):
    bmesh = {}
    if bmesh_type:
        try:
            bmesh_path = get_resource_path(os.path.join('brainmesh'))
            for h in ['lh', 'rh']:
                fpath = os.path.join(bmesh_path, f'{bmesh_type}.{h}.gii')
                if os.path.exists(fpath):
                    bmesh[h] = load_gii2pv(fpath)
        except Exception as e:
            print(f"Warning: Could not load brainmesh. {e}")
    return bmesh
        
def add_context_to_view(plotter, bmesh, view_side, alpha, color, **kwargs):
    """
    Adds context mesh. Lighting parameters are passed via **kwargs.
    """
    if not bmesh: return
    for h, mesh in bmesh.items():
        if (view_side == 'L' and h == 'rh') or (view_side == 'R' and h == 'lh'): continue
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

def finalize_plot(plotter, export_path, display_type):
    if export_path: plotter.screenshot(export_path, transparent_background=True)
    
    if display_type == 'static': plotter.show(jupyter_backend='static')
    elif display_type == 'interactive': plotter.show(jupyter_backend='trame')
    elif display_type == 'none': plotter.close()
    return plotter

