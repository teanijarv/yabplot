# %%
import os
import re
import numpy as np
import nibabel as nib
from skimage import measure
import pandas as pd

# %%
# Desikan

atlas_path = '/Users/to8050an/Documents/other_code/neuroparc-master/atlases/label/Human/Desikan_space-MNI152NLin6_res-2x2x2.nii.gz'
label_path = '/Users/to8050an/Documents/other_code/neuroparc-master/atlases/label/Human/Anatomical-labels-csv/Desikan.csv'
surf_dir = 'yabplot/resources/desikan/gii'
os.makedirs(surf_dir, exist_ok=True)

labels = pd.read_csv(label_path, header=None).rename(columns={0: 'ROI', 1: 'Label'})
rois = {row['ROI']: row['Label'] for _, row in labels.iterrows()}  # Adapt ROI column name as needed

atlas_img = nib.load(atlas_path)
atlas_data = atlas_img.get_fdata()
for roi_id, name in rois.items():
    mask = (atlas_data == roi_id).astype(np.uint8)
    if np.sum(mask) == 0:
        continue
    verts, faces, normals, values = measure.marching_cubes(mask, level=0.5)
    verts_mni = nib.affines.apply_affine(atlas_img.affine, verts)
    coord_array = nib.gifti.GiftiDataArray(data=verts_mni.astype(np.float32), intent='NIFTI_INTENT_POINTSET')
    face_array = nib.gifti.GiftiDataArray(data=faces.astype(np.int32), intent='NIFTI_INTENT_TRIANGLE')
    gii = nib.gifti.GiftiImage(darrays=[coord_array, face_array])
    gii_path = os.path.join(surf_dir, f"{name}_surface.surf.gii")
    nib.save(gii, gii_path)

# %%
# musus100

# dir = get_resource_path(name)
label_path = 'yabplot/resources/atlas/musus100/in/musus100_thalamus.txt'
atlas_path = 'yabplot/resources/atlas/musus100/in/musus100_thalamus.nii.gz'
surf_dir = 'yabplot/resources/atlas/musus100/gii'
os.makedirs(surf_dir, exist_ok=True)

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

atlas_img = nib.load(atlas_path)
atlas_data = atlas_img.get_fdata()
for roi_id, name in rois.items():
    mask = (atlas_data == roi_id).astype(np.uint8)
    if np.sum(mask) == 0:
        continue
    verts, faces, normals, values = measure.marching_cubes(mask, level=0.5)
    verts_mni = nib.affines.apply_affine(atlas_img.affine, verts)
    coord_array = nib.gifti.GiftiDataArray(data=verts_mni.astype(np.float32), intent='NIFTI_INTENT_POINTSET')
    face_array = nib.gifti.GiftiDataArray(data=faces.astype(np.int32), intent='NIFTI_INTENT_TRIANGLE')
    gii = nib.gifti.GiftiImage(darrays=[coord_array, face_array])
    gii_path = os.path.join(surf_dir, f"{name}_surface.surf.gii")
    nib.save(gii, gii_path)

# %%
# xtract

import xml.etree.ElementTree as ET

atlas_path = 'yabplot/resources/xtract/in/xtract-tract-atlases-maxprob5-1mm.nii.gz'
label_path = 'yabplot/resources/xtract/in/XTRACT.xml'
surf_dir = 'yabplot/resources/xtract/gii'
# atlas_path = 'yabplot/resources/jhu/in/JHU-ICBM-labels-1mm.nii.gz'
# label_path = 'yabplot/resources/jhu/in/JHU-labels.xml'
# surf_dir = 'yabplot/resources/jhu/gii'
os.makedirs(surf_dir, exist_ok=True)

rois = {}
tree = ET.parse(label_path)
root = tree.getroot()
for label in root.findall('.//label'):
    roi_id = int(label.get('index'))+1 ## see +1 in here
    name = label.text.strip().replace(' ', '_')
    rois[roi_id] = name

atlas_img = nib.load(atlas_path)
atlas_data = atlas_img.get_fdata()
for roi_id, name in rois.items():
    try:
        mask = (atlas_data == roi_id).astype(np.uint8)
        if np.sum(mask) == 0:
            continue
        verts, faces, normals, values = measure.marching_cubes(mask, level=0.5)
        verts_mni = nib.affines.apply_affine(atlas_img.affine, verts)
        coord_array = nib.gifti.GiftiDataArray(data=verts_mni.astype(np.float32), intent='NIFTI_INTENT_POINTSET')
        face_array = nib.gifti.GiftiDataArray(data=faces.astype(np.int32), intent='NIFTI_INTENT_TRIANGLE')
        gii = nib.gifti.GiftiImage(darrays=[coord_array, face_array])
        gii_path = os.path.join(surf_dir, f"{name}_surface.surf.gii")
        nib.save(gii, gii_path)
    except:
        continue

# %%
# aseg

import re
import nibabel as nib
import numpy as np
import os
from skimage import measure

label_path = 'yabplot/resources/aseg/in/FreeSurferColorLUT.txt'
atlas_path = 'yabplot/resources/aseg/in/aseg.mgz'
surf_dir = 'yabplot/resources/aseg/gii'
os.makedirs(surf_dir, exist_ok=True)

# Parse FreeSurferColorLUT.txt
rois = {}
pattern = re.compile(r'^ *(\d+)\s+([^\s]+)\s+\d+\s+\d+\s+\d+\s+\d+')
with open(label_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        m = pattern.match(line)
        if m:
            roi_id = int(m.group(1))
            name = m.group(2).replace("-", "_")
            rois[roi_id] = name

atlas_img = nib.load(atlas_path)
atlas_data = atlas_img.get_fdata()

for roi_id, name in rois.items():
    mask = (atlas_data == roi_id).astype(np.uint8)
    if np.sum(mask) == 0:
        continue
    verts, faces, normals, values = measure.marching_cubes(mask, level=0.5)
    verts_mni = nib.affines.apply_affine(atlas_img.affine, verts)
    coord_array = nib.gifti.GiftiDataArray(data=verts_mni.astype(np.float32), intent='NIFTI_INTENT_POINTSET')
    face_array = nib.gifti.GiftiDataArray(data=faces.astype(np.int32), intent='NIFTI_INTENT_TRIANGLE')
    gii = nib.gifti.GiftiImage(darrays=[coord_array, face_array])
    gii_path = os.path.join(surf_dir, f"{name}_surface.surf.gii")
    nib.save(gii, gii_path)

# %%

import re
import nibabel as nib
import numpy as np
import os
from skimage import measure

label_path = 'yabplot/resources/brainnetome/in/BN_Atlas_246_LUT.txt'
atlas_path = 'yabplot/resources/brainnetome/in/BN_Atlas_246_1mm.nii.gz'
surf_dir = 'yabplot/resources/brainnetome/gii'
os.makedirs(surf_dir, exist_ok=True)

# Parse BN_Atlas_246_LUT.txt
rois = {}
pattern = re.compile(r'^ *(\d+)\s+([^\s]+)\s+\d+\s+\d+\s+\d+\s+\d+')
with open(label_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue  # Skip empty lines or comments
        m = pattern.match(line)
        if m:
            roi_id = int(m.group(1))
            name = m.group(2).replace("-", "_")
            rois[roi_id] = name

atlas_img = nib.load(atlas_path)
atlas_data = atlas_img.get_fdata()

for roi_id, name in rois.items():
    print(roi_id)
    mask = (atlas_data == roi_id).astype(np.uint8)
    if np.sum(mask) == 0:
        continue
    verts, faces, normals, values = measure.marching_cubes(mask, level=0.5)
    verts_mni = nib.affines.apply_affine(atlas_img.affine, verts)
    coord_array = nib.gifti.GiftiDataArray(data=verts_mni.astype(np.float32), intent='NIFTI_INTENT_POINTSET')
    face_array = nib.gifti.GiftiDataArray(data=faces.astype(np.int32), intent='NIFTI_INTENT_TRIANGLE')
    gii = nib.gifti.GiftiImage(darrays=[coord_array, face_array])
    gii_path = os.path.join(surf_dir, f"{name}_surface.surf.gii")
    nib.save(gii, gii_path)
# %%
