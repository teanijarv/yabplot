# %%
import os
import re
import numpy as np
import nibabel as nib
from skimage import measure
from yabplot import get_resource_path

name = 'musus100_dbn'
verbose = True

# file checks

dir = get_resource_path(name)
label_path = os.path.join(dir, 'DBN.txt')
atlas_path = os.path.join(dir, 'DBN.nii.gz')

surf_dir = os.path.join(dir, 'surfs')
os.makedirs(surf_dir, exist_ok=True)

# %%

# Parse the ITK-Snap label file
rois = {}
pattern = re.compile(r'^ *(\d+)\s+.+?"(.*?)"$')
with open(label_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue  # skip comments/empty lines
        m = pattern.match(line)
        if m:
            roi_id = int(m.group(1))
            name = m.group(2).replace(" ", "_")
            rois[roi_id] = name

if verbose: print("Extracted ROI labels:", rois)

# Extract surface meshes for each atlas label
atlas_img = nib.load(atlas_path)
atlas_data = atlas_img.get_fdata()
for roi_id, name in rois.items():
    mask = (atlas_data == roi_id).astype(np.uint8)
    if np.sum(mask) == 0:
        continue  # skip empty masks

    # Marching cubes extracts mesh from binary volume
    verts, faces, normals, values = measure.marching_cubes(mask, level=0.5)

    # Convert voxel coordinates to MNI coordinates
    verts_mni = nib.affines.apply_affine(atlas_img.affine, verts)

    # Save as GIFTI mesh
    coord_array = nib.gifti.GiftiDataArray(data=verts_mni.astype(np.float32),
        intent='NIFTI_INTENT_POINTSET')
    face_array = nib.gifti.GiftiDataArray(data=faces.astype(np.int32),
        intent='NIFTI_INTENT_TRIANGLE')
    gii = nib.gifti.GiftiImage(darrays=[coord_array, face_array])
    gii_path = os.path.join(surf_dir, f"{name}_surface.surf.gii")
    nib.save(gii, gii_path)
    if verbose: print(f"Saved surface mesh of {name} to {gii_path}")

# %%
