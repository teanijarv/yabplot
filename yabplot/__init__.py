from importlib.metadata import version, PackageNotFoundError

from .plotting import plot_cortical, plot_subcortical, plot_tracts, plot_voxelwise, clear_tract_cache
from .data import get_available_resources, get_atlas_regions, get_surface_paths
from .atlas_builder import build_cortical_atlas, build_subcortical_atlas

try:
    __version__ = version("yabplot")
except PackageNotFoundError:
    __version__ = "unknown"