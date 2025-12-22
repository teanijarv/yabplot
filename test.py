# %%

from yabplot import plot_subcortical



print('subcortical brainnetome-sc atlas, with default lighting and all views:')
p = plot_subcortical(atlas='brainnetome_sc', style='default')

print('subcortical aseg atlas, with sculpted lighting:')
p = plot_subcortical(atlas='aseg', style='sculpted', figsize=(1000, 300),
                     views=['left_lateral', 'left_medial', 'superior', 'anterior'])

print('subcortical aseg atlas with data, with matte lighting:')
data = {'Left_Hippocampus': 0.6, 'Left_Amygdala': -0.2,
        'Left_Putamen': 0.9, 'Left_Thalamus': -0.9}
p = plot_subcortical(data=data, atlas='aseg', style='matte', figsize=(1000, 300),
                     views=['left_lateral', 'left_medial', 'superior', 'anterior'])

print('subcortical aseg atlas with data with modified brainmesh and in 2D:')
p = plot_subcortical(data=data, atlas='aseg', style='flat', figsize=(1000, 300),
                     views=['left_lateral', 'left_medial', 'superior', 'anterior'],
                     bmesh_color='gray', bmesh_alpha=0.05)

# %%

from yabplot import plot_tracts

data = {
    'FMaj': 0.7,
    'FMin': 0.9,
    'CRT_L': 0.9,
    'IFOF_L': -0.5,
    'ILF_L': -0.9,
    'UF_L': 0.2,
    'ATR_L': 0.5,
    'SLF2_L': 0.2,
    'SLF3_L': 0.2,
}
s = 'default'

p = plot_tracts(atlas='xtract_tiny', layout=(2, 2), zoom=1.4, style=s,
                views=['left_lateral', 'left_medial', 'superior', 'anterior'],
                orientation_coloring=True)
p = plot_tracts(atlas='xtract_tiny', layout=(2, 2), zoom=1.4, style=s,
                views=['left_lateral', 'left_medial', 'superior', 'anterior'])
p = plot_tracts(atlas='xtract_tiny', data=data, layout=(2, 2), zoom=1.4, style=s,
                views=['left_lateral', 'left_medial', 'superior', 'anterior'])

# %%
import numpy as np
import pandas as pd
from yabplot import plot_cortical

d_gr = pd.read_csv('/Users/to8050an/Documents/data/margulies_gradients.csv')

p = plot_cortical(
    data=np.array(d_gr['gradient1']), 
    atlas='schaefer_1000'
)

# %%
import numpy as np
import pandas as pd
from yabplot import plot_cortical

s = 'default'

plot_cortical(atlas='schaefer_1000', style=s)
d_gr = pd.read_csv('/Users/to8050an/Documents/data/margulies_gradients.csv')
p = plot_cortical(
    data=np.array(d_gr['gradient1']), 
    atlas='schaefer_1000',
    style=s
)

# %%
