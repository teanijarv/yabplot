from importlib.resources import files
import numpy as np
import nibabel as nib
import pyvista as pv

def get_resource_path(relpath=''):
    resource_file = files('yabplot.resources') / relpath
    return str(resource_file)

def smooth_mesh_vertices(verts, faces, iterations=2, factor=0.5):
    """Laplacian smoothing."""
    verts_smooth = verts.copy()
    
    for _ in range(iterations):
        verts_new = np.zeros_like(verts_smooth)
        counts = np.zeros(len(verts_smooth))
        
        for face in faces:
            for i in range(3):
                for j in range(3):
                    if i != j:
                        verts_new[face[i]] += verts_smooth[face[j]]
                        counts[face[i]] += 1
        
        mask = counts > 0
        verts_new[mask] /= counts[mask][:, np.newaxis]
        verts_smooth = factor * verts_new + (1 - factor) * verts_smooth
    
    return verts_smooth

def load_gii2pv(gii_path, smooth=False):
    """Load GIfTI and convert to pyvista format (+ smooth)."""
    mesh = nib.load(gii_path)
    verts = mesh.darrays[0].data
    faces = mesh.darrays[1].data

    if smooth:
        verts = smooth_mesh_vertices(verts, faces, iterations=15, factor=0.6)

    faces_pv = np.column_stack([np.full(len(faces), 3), faces]).flatten()
    pv_mesh = pv.PolyData(verts, faces_pv)

    return pv_mesh

def prep_data(data):
    """Convert data to dictionary if it is not."""
    if isinstance(data, pd.DataFrame):
        # assume first column is region names, second is values
        if data.shape[1] >= 2:
            data = dict(zip(data.iloc[:, 0], data.iloc[:, 1]))
        else:
            raise ValueError("DataFrame must have at least 2 columns")
    elif isinstance(data, pd.Series):
        data = data.to_dict()
    elif not isinstance(data, dict):
        raise ValueError("Data must be dict, DataFrame, or Series")
    
    return data

def generate_distinct_colors(n_colors, seed=42):
    """Generate visually distinct colors using HSV color space."""
    np.random.seed(seed)
    colors = []
    
    golden_ratio = 0.618033988749895
    hue = np.random.rand()
    
    for i in range(n_colors):
        hue += golden_ratio
        hue %= 1.0
        
        saturation = 0.6 + np.random.rand() * 0.3
        value = 0.7 + np.random.rand() * 0.2
        
        h_i = int(hue * 6)
        f = hue * 6 - h_i
        p = value * (1 - saturation)
        q = value * (1 - f * saturation)
        t = value * (1 - (1 - f) * saturation)
        
        if h_i == 0:
            r, g, b = value, t, p
        elif h_i == 1:
            r, g, b = q, value, p
        elif h_i == 2:
            r, g, b = p, value, t
        elif h_i == 3:
            r, g, b = p, q, value
        elif h_i == 4:
            r, g, b = t, p, value
        else:
            r, g, b = value, p, q
        
        colors.append((r, g, b))
    
    return colors
