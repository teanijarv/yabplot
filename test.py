# %%

from yabplot import plot_roi

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

p = plot_roi(data=data, layout=(1, 3), atlas='jhu', vminmax=[-1, 1], figsize=(1000, 500), zoom=1.3,
             views=['left_lateral', 'left_medial', 'superior'], nan_alpha=0)

# %%

from yabplot import plot_roi

data = {
    'lAmyg_L': 0.7,
    'mAmyg_L': 0.9,
    'cHipp_L': -0.5,
    'rHipp_L': -0.9,
    'Otha_L': 0.2,
    # 'Body_of_corpus_callosum': 0.5,
    # 'Posterior_corona_radiata_R': 0.2,
}

p = plot_roi(data=data, layout=(1, 3), atlas='brainnetome', vminmax=[-1, 1], figsize=(1000, 500), zoom=1.3,
             views=['left_lateral', 'left_medial', 'superior'], nan_alpha=1)

# %%

p = plot_roi(data=None, layout=(1, 3), atlas='jhu', vminmax=[-1, 1], figsize=(650, 300), zoom=1.3,
             views=['left_lateral', 'left_medial', 'superior'], nan_alpha=1, display_type='static')
# %%


# %%
