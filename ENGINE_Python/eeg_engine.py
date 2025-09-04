from eeg_positions import get_elec_coords, plot_coords
import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.pyplot import savefig

df = pd.read_csv('/Users/laurensidelinger/PycharmProjects/ENGINE/regression_bandmeasurecombo/Gamma_Abs_Power$_interpretationmodel_features.csv')
df['Electrode'] = df['Feature_Name'].str.split('_').str[0]

chans = """Fp2 F3 Fz F4 Cz P7 P3 P4 P8 O1 O2 Fp1 F7 F8 T7 C3 C4 T8 Pz""".split()


whitechannels = list(set(chans) - set(df['Electrode']))
whites = pd.DataFrame(whitechannels, columns = ['Electrode'])
whites['Importance_Score'] = [0] * len(whitechannels)
all = pd.concat([df, whites])

all = all.sort_values(by='Importance_Score')
importances = all['Importance_Score']


norm = Normalize(vmin=importances.min(), vmax=importances.max())
cmap = cm.Purples

colors_hex = importances.apply(lambda x: cmap(norm(x)))
all['colors_hex'] = colors_hex.apply(lambda rgba: matplotlib.colors.rgb2hex(rgba))
print(all)

channels = list(all['Electrode'])
print(channels)

coords=get_elec_coords(elec_names=channels)
print(coords)


colors = (
    list(all['colors_hex'])
)
print(colors)

colors_font = (
    ["white"] * 5,
    ["black"] * 14
)

plt.rcParams['font.family'] = 'Cambria'

fig, ax = plot_coords(
    coords,
    scatter_kwargs={
        "s":1200,
        "color": colors,
        "linewidths":2
    },
    text_kwargs={
        "color": "white",
        "fontsize": 20,
        "ha": "center",
        "va": "center"
    }
)

for text in ax.texts:
    text.set_path_effects([
        path_effects.Stroke(linewidth=5, foreground="black"),
        path_effects.Normal()
    ])

#plt.title('Beta Absolute Power Important Features', fontsize=20)
savefig('gammaeegimportance.png', transparent = True)