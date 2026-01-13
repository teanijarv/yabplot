from importlib.metadata import version, PackageNotFoundError

from .plotting import plot_cortical, plot_subcortical, plot_tracts, clear_tract_cache
from .data import get_available_resources, get_atlas_regions

try:
    __version__ = version("yabplot")
except PackageNotFoundError:
    __version__ = "unknown"