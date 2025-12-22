# %%

from yabplot import plot_subcortical

# import numpy as np
# data = {'L-SNr(substantia_nigra_pars_reticulata)': np.float64(1),
#  'L-SNc(substantia_nigra_pars_compacta)': np.float64(1)}

data = {
    'Left_Hippocampus': 0.7,
    'Left_Amygdala': -0.5,
    'Genu_of_corpus_callosum': 0.7,
    'Body_of_corpus_callosum': 0,
    # 'Posterior_corona_radiata_R': 0.2,
}

p = plot_subcortical(data=None, layout=(1, 3), atlas='jhu', vminmax=[-1, 1], figsize=(1000, 500), zoom=1.3,
             views=['left_lateral', 'left_medial', 'superior'], nan_alpha=0, legend=True)

# %%

from yabplot import plot_tracts

data = {
    'Forceps_Major': 0.7,
    'Forceps_Minor': 0.9,
    'Corticospinal_Tract_L': 0.9,
    'Inferior_Fronto-Occipital_Fasciculus_L': -0.5,
    'Inferior_Longitudinal_Fasciculus_L': -0.9,
    'Uncinate_Fasciculus_L': 0.2,
    'Anterior_Thalamic_Radiation_L': 0.5,
    'Superior_Longitudinal_Fasciculus_2_L': 0.2,
    'Superior_Longitudinal_Fasciculus_3_L': 0.2,
}

p = plot_tracts(data=None, layout=(1, 2), atlas='xtract_tiny', vminmax=[-1, 1], figsize=(1000, 500), zoom=1.3,
             views=['left_lateral', 'superior'], nan_alpha=1, bmesh_type='fsaverage')

# %%

p = plot_subcortical(data=None, layout=(1, 3), atlas='aseg', vminmax=[-1, 1], figsize=(650, 300), zoom=1.3,
             views=['left_lateral', 'left_medial', 'superior'], nan_alpha=1)
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
