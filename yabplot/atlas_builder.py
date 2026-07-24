import os
import glob
import numpy as np
import nibabel as nib
import scipy.sparse as sp
import pyvista as pv
from skimage import measure
from .wrappers import run_wb_import, run_wb_projection
from .data import get_surface_paths
from .plotting import plot_cortical, plot_subcortical

### CORTICAL

def _build_adjacency(surf_path, n_vert=32492):
    """internal helper to build surface adjacency matrix."""
    surf = nib.load(surf_path)
    faces = surf.darrays[1].data.astype(int)
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    row, col = np.concatenate([edges[:, 0], edges[:, 1]]), np.concatenate([edges[:, 1], edges[:, 0]])
    return sp.coo_matrix((np.ones(len(row), dtype=int), (row, col)), shape=(n_vert, n_vert)).tocsr()

def build_cortical_atlas(nii_path, wb_txt_path, out_dir, include_list=None, exclude_list=None, atlasname='atlas'):
    """
    builds a custom yabplot cortical atlas from a volumetric NIfTI file.

    projects a volumetric NIfTI atlas to standard fsLR32k surfaces using connectome workbench, 
    cleans the medial wall, and applies majority-vote boundary smoothing to remove voxel artifacts.

    parameters
    ----------
    nii_path : str
        absolute path to the 3D NIfTI volume of the atlas.
    wb_txt_path : str
        absolute path to the text file formatted specifically for connectome workbench.
    out_dir : str
        directory where the final .csv map and .txt LUT will be saved.
    include_list : list of str, optional
        keywords of regions to strictly include. all other regions are ignored.
    exclude_list : list of str, optional
        keywords of regions to strictly exclude. all other regions are kept.
    atlasname : str, optional
        prefix name for the output files. default is 'atlas'.

    raises
    ------
    ValueError
        if both include_list and exclude_list are provided.
    """
    if include_list and exclude_list:
        raise ValueError("please provide either 'include_list' or 'exclude_list', not both.")

    os.makedirs(out_dir, exist_ok=True)
    
    # define intermediate and output paths
    labeled_nii = os.path.join(out_dir, 'temp_labeled.nii.gz')
    lh_gii = os.path.join(out_dir, 'lh_temp.label.gii')
    rh_gii = os.path.join(out_dir, 'rh_temp.label.gii')
    out_csv = os.path.join(out_dir, f'{atlasname}.csv')
    out_lut = os.path.join(out_dir, f'{atlasname}.txt')

    # fetch standard fsLR32k surfaces and masks via yabplot data system
    print("fetching standard surfaces...")
    lh_mid, rh_mid = get_surface_paths('midthickness', 'bmesh')
    lh_white, rh_white = get_surface_paths('white', 'bmesh')
    lh_pial, rh_pial = get_surface_paths('pial', 'bmesh')
    lh_mask_path, rh_mask_path = get_surface_paths('nomedialwall', 'label')

    # run wb_command wrappers
    print("running volume-to-surface projection...")
    run_wb_import(nii_path, wb_txt_path, labeled_nii)
    run_wb_projection(labeled_nii, lh_mid, lh_gii, lh_white, lh_pial)
    run_wb_projection(labeled_nii, rh_mid, rh_gii, rh_white, rh_pial)

    # extract LUT and apply include/exclude filtering
    labels_dict = nib.load(lh_gii).labeltable.get_labels_as_dict()
    valid_ids = []
    lut_dict = {} 
    
    for rid, name in labels_dict.items():
        if rid == 0 or name == '???': continue 
        
        # filter logic
        if include_list:
            if not any(inc in name for inc in include_list):
                continue
        elif exclude_list:
            if any(exc in name for exc in exclude_list):
                continue
            
        clean_name = name.replace(' ', '_').replace('/', '-')
        np.random.seed(rid)
        r, g, b = np.random.randint(50, 255, 3) 
        
        # store the string in the dictionary instead of writing to a file yet
        lut_dict[rid] = f"{rid}  {clean_name}  {r}  {g}  {b}  0"
        valid_ids.append(rid)
        
    print(f"found {len(valid_ids)} initial cortical regions. mapping and cleaning...")

    # merge LH and RH, then apply masks
    data = np.concatenate([
        nib.load(lh_gii).darrays[0].data.astype(int).flatten(),
        nib.load(rh_gii).darrays[0].data.astype(int).flatten()
    ])
    
    mask = np.concatenate([
        nib.load(lh_mask_path).darrays[0].data.astype(int).flatten() != 0,
        nib.load(rh_mask_path).darrays[0].data.astype(int).flatten() != 0
    ])

    data[~mask] = 0
    data[~np.isin(data, valid_ids)] = 0

    # build adjacency and run hole-filling
    print("building surface adjacency and filling holes...")
    adj = sp.block_diag((_build_adjacency(lh_mid), _build_adjacency(rh_mid))).tocsr()
    adj.setdiag(1) 
    n_vert = len(data)

    for _ in range(20): 
        holes = (data == 0) & mask
        if not np.any(holes): break
        
        unique, inv = np.unique(data, return_inverse=True)
        one_hot = sp.coo_matrix((np.ones(n_vert), (np.arange(n_vert), inv)), shape=(n_vert, len(unique))).tocsr()
        votes = (adj @ one_hot).toarray()
        
        zero_idx = np.where(unique == 0)[0][0]
        votes[:, zero_idx] = 0 
        
        winner = np.argmax(votes, axis=1)
        fill_vals = unique[winner]
        data[holes] = fill_vals[holes]

    # smooth final boundaries
    print("smoothing boundaries...")
    for _ in range(10): 
        unique, inv = np.unique(data, return_inverse=True)
        one_hot = sp.coo_matrix((np.ones(n_vert), (np.arange(n_vert), inv)), shape=(n_vert, len(unique))).tocsr()
        winner = np.argmax((adj @ one_hot).toarray(), axis=1)
        data = unique[winner]
        data[~mask] = 0

    # save the final vertex map
    np.savetxt(out_csv, data, fmt='%i')
    
    # find out which regions actually survived the smoothing/masking
    surviving_ids = np.unique(data)
    
    # filter the LUT lines to only include survivors
    final_lines = []
    dropped_count = 0
    
    for rid, line_str in lut_dict.items():
        if rid in surviving_ids:
            final_lines.append(line_str)
        else:
            region_name = line_str.split()[1]
            print(f"[WARNING] {region_name} (id {rid}) lost during smoothing/masking. dropping from lut.")
            dropped_count += 1
            
    # write the perfectly clean file
    with open(out_lut, 'w') as f:
        f.write("\n".join(final_lines))

    print(f"final polished atlas saved to: {out_dir}")
    print(f"saved {len(final_lines)} regions ({dropped_count} empty regions dropped).")
    
    # cleanup intermediate Workbench files to save space
    for temp_file in [labeled_nii, lh_gii, rh_gii]:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def qc_custom_cortical_atlas(atlas_dir, atlasname='atlas'):
    """
    generates a quality control report for a custom cortical atlas.
    
    reads the generated vertex map and lookup table, counts the vertices for each region, 
    saves a summary text file, and generates individual static plots for every region 
    to help identify mapping dropouts or anatomical bleed.

    parameters
    ----------
    atlas_dir : str
        absolute path to the custom atlas directory containing the .csv and .txt files.
    atlasname : str, optional
        prefix name of the files to check. default is 'atlas'.
    """
    
    csv_path = os.path.join(atlas_dir, f"{atlasname}.csv")
    lut_path = os.path.join(atlas_dir, f"{atlasname}.txt")
    qc_dir = os.path.join(atlas_dir, "qc_report")
    
    os.makedirs(qc_dir, exist_ok=True)
    
    # load the mapped data and the lookup table
    labels = np.loadtxt(csv_path).astype(int)
    
    regions = {}
    with open(lut_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                regions[int(parts[0])] = parts[1]
                
    print(f"starting qc for {len(regions)} regions...\n")
    
    report_path = os.path.join(qc_dir, "_vertex_counts.txt")
    
    with open(report_path, 'w') as f_out:
        f_out.write("region_name\tid\tvertex_count\n")
        f_out.write("-" * 40 + "\n")
        
        for rid, name in regions.items():
            count = np.sum(labels == rid)
            f_out.write(f"{name}\t{rid}\t{count}\n")
            
            print(f"[{name}] id: {rid} | vertices: {count}")
            
            if count == 0:
                print(f"[WARNING] {name} is empty! skipping plot.")
                continue
            
            plot_file = os.path.join(qc_dir, f"{rid:03d}_{name}.png")
            
            try:
                ax = plot_cortical(
                    data={name: 1},
                    custom_atlas_path=atlas_dir,
                    cmap='binary', vminmax=[0, 1],
                    export_path=plot_file
                )
            except Exception as e:
                print(f"  -> failed to plot {name}: {e}")
                
    print(f"\nqc complete! check the '{qc_dir}' folder for the report and images.")


### SUBCORTICAL

def build_subcortical_atlas(nii_path, labels_dict, out_dir, include_list=None, exclude_list=None,
                            smooth_i=15, smooth_f=0.6):
    """
    extracts 3D subcortical meshes from a volumetric nifti atlas.

    uses the marching cubes algorithm to generate 3D surface meshes for specific 
    regions, applies laplacian smoothing to remove voxel artifacts, and saves them as .vtk files.

    parameters
    ----------
    nii_path : str
        absolute path to the 3D nifti volume.
    labels_dict : dict
        dictionary mapping integer region IDs to string names (e.g., {1: 'thalamus_l'}).
    out_dir : str
        directory where the .vtk mesh files will be saved.
    include_list : list of str, optional
        keywords of regions to strictly include. all other regions are ignored.
    exclude_list : list of str, optional
        keywords of regions to strictly exclude. all other regions are kept.
    smooth_i : int, optional
        number of iterations for laplacian mesh smoothing. default is 15.
    smooth_f : float, optional
        relaxation factor for laplacian mesh smoothing (0.0 to 1.0). default is 0.6.

    raises
    ------
    ValueError
        if both include_list and exclude_list are provided.
    """
    if include_list and exclude_list:
        raise ValueError("please provide either 'include_list' or 'exclude_list', not both.")
        
    os.makedirs(out_dir, exist_ok=True)
    
    # apply the include/exclude filters to the provided dictionary
    targets = {}
    for rid, name in labels_dict.items():
        if include_list:
            if not any(inc in name for inc in include_list):
                continue
        elif exclude_list:
            if any(exc in name for exc in exclude_list):
                continue
                
        targets[rid] = name
                    
    print(f"filtered down to {len(targets)} subcortical regions to extract.")

    # load the nifti volume and its affine matrix
    img = nib.load(nii_path)
    data = img.get_fdata().round()
    affine = img.affine

    # loop through targets, extract meshes, and save
    for rid, name in targets.items():
        # create a binary mask for just this region
        mask = (data == rid).astype(np.uint8)
        
        # skip if empty
        if np.sum(mask) == 0:
            print(f"[WARNING] {name} is empty in the volume!")
            continue

        print(f"extracting: {name} (id {rid})...")
        
        # run marching cubes to get raw vertices and faces
        verts, faces, normals, values = measure.marching_cubes(mask, level=0.5)

        # apply the nifti affine matrix cleanly using nibabel
        verts_mni = nib.affines.apply_affine(affine, verts)

        # format faces for pyvista: [n_points, p1, p2, p3, n_points, p1, p2, p3...]
        faces_pv = np.column_stack((np.full(len(faces), 3), faces)).flatten()
        
        # create the 3d pyvista mesh
        mesh = pv.PolyData(verts_mni, faces_pv)
        
        # apply laplacian smoothing to melt away the blocky voxel edges
        mesh = mesh.smooth(n_iter=smooth_i, relaxation_factor=smooth_f)
        mesh.compute_normals(inplace=True)

        # we remove super small structures which would not be visible
        if mesh.n_points < 4 or abs(mesh.volume) < 0.01:
            print(f"[WARNING] {name} is too small to form a 3D mesh (volume: {abs(mesh.volume):.4f} mm³). dropping from atlas.")
            continue
        
        # save as a vtk file
        out_file = os.path.join(out_dir, f"{name}.vtk")
        mesh.save(out_file)
    
    # file for ordering the labels
    lut_path = os.path.join(out_dir, "atlas_LUT.txt")
    with open(lut_path, 'w') as f:
        for rid, name in targets.items():
            f.write(f"{rid} {name}\n")

    print(f"\nsubcortical atlas successfully saved to: {out_dir}")

def qc_custom_subcortical_atlas(atlas_dir):
    """
    generates a quality control report for a custom subcortical atlas.
    
    reads the generated .vtk meshes, calculates their geometric properties 
    (vertices, faces, volume), saves a summary text file, and generates 
    individual static plots for every region to help identify corrupt meshes 
    or anatomical artifacts.

    parameters
    ----------
    atlas_dir : str
        absolute path to the custom atlas directory containing the .vtk files.
    """
    
    qc_dir = os.path.join(atlas_dir, "qc_report")
    os.makedirs(qc_dir, exist_ok=True)
    
    # find all vtk files in the atlas directory
    vtk_files = glob.glob(os.path.join(atlas_dir, "*.vtk"))
    
    if not vtk_files:
        print(f"no .vtk files found in {atlas_dir}. cannot run qc.")
        return
        
    print(f"starting qc for {len(vtk_files)} subcortical meshes...\n")
    
    report_path = os.path.join(qc_dir, "_mesh_properties.txt")
    
    with open(report_path, 'w') as f_out:
        # header for our text report
        f_out.write("region_name\tvertices\tfaces\tvolume_mm3\n")
        f_out.write("-" * 55 + "\n")
        
        for vtk_path in sorted(vtk_files):
            filename = os.path.basename(vtk_path)
            region_name = os.path.splitext(filename)[0]
            
            # read mesh to extract physical properties
            try:
                mesh = pv.read(vtk_path)
                n_verts = mesh.n_points
                n_faces = mesh.n_cells
                volume = mesh.volume
            except Exception as e:
                print(f"  -> error reading {filename}: {e}")
                f_out.write(f"{region_name}\tERROR\tERROR\tERROR\n")
                continue
                
            f_out.write(f"{region_name}\t{n_verts}\t{n_faces}\t{volume:.2f}\n")
            print(f"[{region_name}] vertices: {n_verts} | volume: {volume:.1f} mm³")
            
            # check for empty or severely corrupted meshes
            if n_verts == 0:
                print(f"[WARNING] {region_name} mesh is empty! skipping plot.")
                continue
            
            plot_file = os.path.join(qc_dir, f"{region_name}.png")
            
            try:
                ax = plot_subcortical(
                    data={region_name: 1},
                    custom_atlas_path=atlas_dir,
                    cmap='binary', vminmax=[0, 1],
                    nan_alpha=0.2,
                    export_path=plot_file
                )
            except Exception as e:
                print(f"  -> failed to plot {region_name}: {e}")
                
    print(f"\nqc complete! check the '{qc_dir}' folder for the report and images.")