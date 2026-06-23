# %%
#%%
import os
if os.getcwd().endswith('notebooks'):
    os.chdir('..')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

from soil_diskin.data_wrangling import assign_biome_numpy

# %%
biome_data = pd.read_csv('data/whittaker_biomes/biomes.csv')
biome_data['y'] = biome_data['y'] * 10 # convert to cm

biome_colors = pd.Series(("NavajoWhite3",
                "DarkGoldenrod1",
                "sienna",
                "DarkOliveGreen4",
                "DarkSeaGreen3",
                "ForestGreen",
                "DarkGreen",
                "OliveDrab",
                "grey"), index=biome_data.biome.unique(),name='color')
biome_data = biome_data.join(biome_colors, on='biome', how='left') 
# biome_data['y'] = biome_data['y'] * 10 # convert to cm

colors = {
    'NavajoWhite3': '#CDB38B',
    'DarkGoldenrod1': '#FFB90F',
    'sienna': '#A0522D',
    'DarkOliveGreen4': '#6E8B3D',
    'DarkSeaGreen3': '#9BCD9B',
    'ForestGreen': '#228B22',
    'DarkGreen': '#006400',
    'OliveDrab': '#6B8E23',
    'grey': '#BEBEBE',
}


# %%
site_data = pd.read_excel('data/balesdent_2018/balesdent_2018_raw.xlsx', skiprows=7)
site_data['biome'] = site_data.apply(assign_biome_numpy, axis=1, biome_data=biome_data)

fig, axs = plt.subplots(1,3, figsize=(12,4), dpi=300, constrained_layout=True)

site_data['Duration_labeling'].plot.hist(ax=axs[0], bins=np.logspace(0,3,int(np.sqrt(site_data.shape[0]))), edgecolor='black', facecolor='lightgrey')
axs[0].set_xscale('log')
axs[0].set(xlabel='Time since transition [years]', xticks=[1,10,100,1000],  xticklabels=[1,10,100,1000], ylabel='Number of sites')


site_data['Sampling date'].plot.hist(ax=axs[1], bins=20, edgecolor='black', facecolor='lightgrey', ylabel='Number of sites')

# Whittaker diagram
ax = axs[2]
ax.set(xlim=(-16, 30), ylim=(0, 500))
for biome, group in biome_data.groupby('biome'):
    ax.add_patch(patches.Polygon(group[['x', 'y']].values, closed=True, fill=True, facecolor=colors[group['color'].values[0]], label=biome, lw=2, edgecolor='w'))

ax.scatter(site_data['MAT_C'], site_data['PANN_mm']/10, color='black', s=30, label='Data Points', )  # MAP converted to cm
ax.set_xlabel("Mean Annual Temperature (°C)")
ax.set_ylabel("Mean Annual Precipitation (cm)")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# remove right and top spines
for ax in axs:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
x,y = [.1,1.0]
axs[0].text(x, y, 'A', transform=axs[0].transAxes, fontsize=14, va='top', ha='right')
axs[1].text(x, y, 'B', transform=axs[1].transAxes, fontsize=14, va='top', ha='right')
axs[2].text(x, y, 'C', transform=axs[2].transAxes, fontsize=14, va='top', ha='right')


plt.savefig('figures/figS1.png', dpi=300)


