import pytest
import numpy as np
import pandas as pd
from yabplot.mesh import map_values_to_surface
from yabplot.utils import prep_data

def test_prep_data_array():
    """Verify that 1D arrays are correctly mapped to a dictionary by position."""
    regions = ['A', 'B', 'C']
    data = [1, 2, 3]
    result = prep_data(data, regions, 'test_atlas', 'cortical')
    assert isinstance(result, dict)
    assert result['A'] == 1
    assert result['C'] == 3

def test_prep_data_dict():
    """Verify that dictionaries are passed through correctly."""
    regions = ['A', 'B', 'C']
    data = {'A': 10, 'B': 20}
    result = prep_data(data, regions, 'test_atlas', 'cortical')
    assert isinstance(result, dict)
    assert result['A'] == 10
    assert result['B'] == 20

def test_prep_data_series():
    """Verify that pandas Series are correctly converted using their index."""
    regions = ['A', 'B', 'C']
    data = pd.Series([100, 200], index=['A', 'C'])
    result = prep_data(data, regions, 'test_atlas', 'cortical')
    assert result['A'] == 100
    assert result['C'] == 200

def test_prep_data_dataframe():
    """Verify that pandas DataFrames are correctly handled by index or column mapping."""
    regions = ['A', 'B', 'C']
    # 1-col dataframe will use index as labels
    data_1col = pd.DataFrame({'val': [5, 6]}, index=['B', 'A'])
    result_1col = prep_data(data_1col, regions, 'test_atlas', 'cortical')
    assert result_1col['A'] == 6
    assert result_1col['B'] == 5

    # 2-col dataframe will use first column as labels and second as values
    df2 = pd.DataFrame({'Region': ['B', 'A'], 'Value': [50, 60]})
    result_2col = prep_data(df2, regions, 'test_atlas', 'cortical')
    assert result_2col['A'] == 60
    assert result_2col['B'] == 50

def test_prep_data_length_mismatch():
    """Verify that an array length mismatch raises a ValueError."""
    regions = ['A', 'B', 'C']
    data = [1, 2] # length mismatch
    with pytest.raises(ValueError, match="Data length mismatch"):
        prep_data(data, regions, 'test_atlas', 'cortical')

def test_map_values_to_surface():
    """Verify that atlas scalar mapping correctly handles data injection and NaNs."""
    target_labels = np.array([0, 1, 2, 1, 0, 3])
    lut_ids = np.array([0, 1, 2, 3])
    dense_lut_names = ['Region0', 'Region1', 'Region2', 'Region3']

    # dict input
    data_dict = {'Region1': 10.0, 'Region2': 20.0}
    res_dict = map_values_to_surface(data_dict, target_labels, lut_ids, dense_lut_names)
    assert np.isnan(res_dict[0]) # region 0 mapped to NaN
    assert res_dict[1] == 10.0   # region 1 mapped to 10
    assert res_dict[2] == 20.0   # region 2 mapped to 20
    assert np.isnan(res_dict[5]) # target 3 -> region 3 mapped to NaN

    # array input
    data_arr = [0.0, 1.0, 2.0, 3.0]
    res_arr = map_values_to_surface(data_arr, target_labels, lut_ids, dense_lut_names)
    assert res_arr[0] == 0.0
    assert res_arr[1] == 1.0
    assert res_arr[5] == 3.0

    # mismatch length array
    with pytest.raises(ValueError):
        map_values_to_surface([1.0, 2.0], target_labels, lut_ids, dense_lut_names)


def _write_tsf(path, values):
    """Write a minimal MRtrix3 float32 little-endian .tsf file."""
    header = b"mrtrix track scalars\ndatatype: Float32LE\nfile: . 64\nEND\n"
    with open(path, "wb") as fh:
        fh.write(header.ljust(64, b" "))
        np.asarray(values, dtype="<f4").tofile(fh)


def test_prep_data_loads_tsf_paths(tmp_path):
    """Verify that .tsf file paths passed to prep_data are loaded via load_tsf."""
    tsf_path = tmp_path / "tract_a.tsf"
    _write_tsf(tsf_path, [0.1, 0.2, np.nan, 0.3, np.inf])

    data_dict = {"A": str(tsf_path)}
    result = prep_data(data_dict, ["A", "B"], "test_atlas", "tracts")
    assert np.allclose(result["A"], [0.1, 0.2, 0.3])
    assert data_dict["A"] == str(tsf_path)

    result_list = prep_data([str(tsf_path)], ["A"], "test_atlas", "tracts")
    assert np.allclose(result_list["A"], [0.1, 0.2, 0.3])


def test_plot_tracts_with_tsf_file(tmp_path):
    """Verify that plot_tracts accepts a .tsf file path in data."""
    import nibabel as nib
    import yabplot as yab

    atlas_dir = tmp_path / "tracts_atlas"
    atlas_dir.mkdir()
    streamlines = [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32),
    ]
    tractogram = nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4))
    nib.streamlines.save(tractogram, str(atlas_dir / "tract_l.trk"))

    tsf_path = tmp_path / "tract_l.tsf"
    _write_tsf(tsf_path, [0.1, 0.2, 0.3, np.nan, 0.4, 0.5, np.inf])

    plotter = yab.plot_tracts(
        data={"tract_l": str(tsf_path)},
        custom_atlas_path=str(atlas_dir),
        bmesh=None,
        views=["left_lateral"],
        display_type="object",
    )
    plotter.close()
