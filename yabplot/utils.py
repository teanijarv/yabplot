import numpy as np
import pandas as pd
import nibabel as nib
import pyvista as pv
import matplotlib.pyplot as plt
import os

def load_gii(gii_path):
    """Load GIfTI geometry (vertices, faces)."""
    mesh = nib.load(gii_path)
    verts = mesh.darrays[0].data
    faces = mesh.darrays[1].data
    return verts, faces

def load_gii2pv(gii_path, smooth_i=0, smooth_f=0.1):
    """
    Load GIfTI and convert to PyVista format with optional smoothing.

    Parameters
    ----------
    smooth_i : int
        Number of smoothing iterations (e.g. 15).
    smooth_f : float
        Relaxation factor (0.0 to 1.0, e.g. 0.6).
    """
    verts, faces = load_gii(gii_path)

    # create pyvista mesh
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).flatten().astype(int)
    mesh = pv.PolyData(verts, faces_pv)

    # apply smoothing
    if smooth_i > 0:
        # use Laplacian smoothing (standard vtkSmoothPolyDataFilter)
        # note: higher relaxation factors can shrink the mesh significantly
        # if shrinkage is an issue, could consider mesh.smooth_taubin() instead
        mesh = mesh.smooth(n_iter=smooth_i, relaxation_factor=smooth_f)

    return mesh

def array_to_gifti(arr, out_path):
    """Save a 1D numpy array as a GIFTI metric file.

    Parameters
    ----------
    arr : np.ndarray
        1D array of shape (n_vertices,).
    out_path : str
        Output path, e.g. 'input.L.func.gii'.
    """
    darray = nib.gifti.GiftiDataArray(arr.astype(np.float32))
    img = nib.gifti.GiftiImage(darrays=[darray])
    nib.save(img, out_path)

def prep_data(data, regions, atlas, category):
    """Standardize input data to dictionary."""
    if isinstance(data, pd.DataFrame):
        if data.shape[1] >= 2:
            data = dict(zip(data.iloc[:, 0], data.iloc[:, 1]))
        else:
            data = data.iloc[:, 0].to_dict()
    elif isinstance(data, pd.Series):
        data = data.to_dict()
    elif isinstance(data, (list, np.ndarray, tuple)):
        if len(data) != len(regions):
            raise ValueError(
                f"Data length mismatch! Atlas '{atlas}' has {len(regions)} regions, "
                f"but input data has {len(data)}. "
                f"For partial data, use a dictionary, pd.Series, or pd.DataFrame. "
                f"Use `yabplot.get_atlas_regions('{atlas}', '{category}')` to see expected order."
            )
        # map strictly by order
        data = dict(zip(regions, data))

    #resolve any present tsf paths:
    if isinstance(data, dict):
        data = {
            key: load_tsf(value) if isinstance(value, str) else value
            for key, value in data.items()
        }

    return data

def generate_distinct_colors(n_colors, seed=42):
    """Generate visually distinct colors using Golden Ratio."""
    np.random.seed(seed)
    colors = []
    hue = np.random.rand()
    for _ in range(n_colors):
        hue = (hue + 0.618033988749895) % 1.0
        colors.append(plt.cm.hsv(hue)[:3])
    return colors

def parse_lut(lut_path):
    """parses LUT to color array and name list."""

    # load and sort by ID to ensure strict order (1..N)
    df = pd.read_csv(lut_path, sep=r'\s+', header=None)
    df = df.sort_values(by=0)

    ids = df[0].values
    names = df[1].tolist()
    rgb = df.iloc[:, 2:5].values / 255.0

    max_id = ids.max()

    lut_colors = np.full((max_id + 1, 3), 0.5)
    lut_names_list = ["Unknown"] * (max_id + 1)

    lut_colors[ids] = rgb
    for idx, name in zip(ids, names):
        lut_names_list[idx] = name

    return ids, lut_colors, lut_names_list, max_id


def load_tsf(tsf_path: str) -> np.ndarray:
    """
    Reads an MRtrix3 .tsf (track scalar file). Useful for users who
    have already computed tractometry metrics using MRtrix3's `tcksample`
    command and want to plot the resulting values in yabplot.

    Parameters
    ----------
    tsf_path : str
        absolute path to the .tsf file.

    Returns
    -------
    numpy.ndarray
        1D array of scalar values for the streamlines.
    """
    if not os.path.isfile(tsf_path):
        raise FileNotFoundError(f"File not found: {tsf_path}")

    header: dict[str, str] = {}
    data_offset: int | None = None

    with open(tsf_path, "rb") as fh:
        # first line must be the magic string
        magic_line = fh.readline().decode("ascii", errors="replace").strip()
        if not magic_line.lower().startswith("mrtrix track scalars"):
            raise ValueError(
                "Not a valid MRtrix TSF file (missing 'mrtrix track scalars' magic)."
            )
        header["magic"] = magic_line

        while True:
            line = fh.readline()
            if not line:
                raise ValueError("Unexpected end of file while reading header.")
            line = line.decode("ascii", errors="replace").strip()
            if line == "END":
                break

            # parse "key: value" pairs
            colon_pos = line.find(":")
            if colon_pos > 0:
                key = line[:colon_pos].strip()
                value = line[colon_pos + 1 :].strip()
                header[key] = value

                # capture the data offset
                if key.lower() == "file":
                    parts = value.split()
                    data_offset = int(parts[-1])

        if data_offset is None:
            raise ValueError("Could not determine data offset from header.")

        # read the binary data
        fh.seek(data_offset)
        raw_bytes = fh.read()

    # determine byte order from header
    datatype = header.get("datatype", "Float32LE").lower()
    byte_order = ">" if datatype.endswith("be") else "<"

    if "64" in datatype:
        dtype = np.dtype(f"{byte_order}f8")
    else:
        dtype = np.dtype(f"{byte_order}f4")

    # trim any trailing bytes that don't fill a complete element
    element_size = dtype.itemsize
    usable = len(raw_bytes) - (len(raw_bytes) % element_size)
    raw_data = np.frombuffer(raw_bytes[:usable], dtype=dtype)

    # split into per-streamline vectors
    inf_mask = np.isinf(raw_data)
    inf_indices = np.where(inf_mask)[0]
    if inf_indices.size > 0:
        raw_data = raw_data[: inf_indices[0]]

    nan_mask = np.isnan(raw_data)
    return raw_data[~nan_mask]
