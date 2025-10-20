# %%
import numpy as np
import matplotlib.pyplot as plt
import itertools as it

import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

import viz 

# use the style file
plt.style.use('notebooks/style.mpl')

colors = viz.color_palette()
color_name_order = np.array([k for k, _ in colors.items()])
np.random.shuffle(color_name_order)
#color_name_order = 'dark_blue,light_green,purple,yellow,red,light_blue,pale_green,dark_purple,yellow,light_red'.split(',')
color_order = [colors[n] for n in color_name_order]

# %%
# input
np.random.seed(1234)
J_t = np.random.normal(10, 2, 2000)

# tri-exponential decay in age
def decay(G_0, ages):
    ks = np.array([0.1, 0.05, 0.01])
    residual = G_0 * np.exp(-np.sum(ks) * ages)
    return residual


# %%
mosaic = 'AABDDEE\nAACDDEE'
fig, axs = plt.subplot_mosaic(mosaic, layout='constrained', figsize=(14, 3), dpi=300)


ax = axs['A']

ax.set_xlim(0, 10)
ax.set_ylim(3, 10)
ax.axis('off') # Hide the axes

# add label (a)
ax.text(1, 10, 'A', ha='center', va='center', fontsize=14)

# 1. Define and draw the boxes for the pools
fast_pool_rect = patches.Rectangle((3.1, 7.5), 3.8, 1, edgecolor='black', facecolor='white', lw=1.5)
slow_pool_rect = patches.Rectangle((3.1, 5.5), 3.8, 1, edgecolor='black', facecolor='white', lw=1.5)
passive_rect = patches.Rectangle((3.1, 3.5), 3.8, 1, edgecolor='black', facecolor='white', lw=1.5)

ax.add_patch(fast_pool_rect)
ax.add_patch(slow_pool_rect)
ax.add_patch(passive_rect)

# 2. Add text labels inside the boxes
ax.text(5, 8, 'fast pool', ha='center', va='center', fontsize=14)
ax.text(5, 6, 'slow pool', ha='center', va='center', fontsize=14)
ax.text(5, 4, 'passive', ha='center', va='center', fontsize=14)

# 3. Add the "input carbon" label
ax.text(1, 7, 'input\ncarbon', ha='center', va='center', fontsize=14)

# 4. Add all the arrows
# Dotted arrows for input carbon with solid arrowheads
# First arrow: draw dotted line, then solid arrowhead
ax.plot([2.15, 2.85], [7.3, 7.86], 'k:', linewidth=2)  # dotted line
ax.annotate('', xy=(3, 8), xytext=(2.85, 7.85), 
            arrowprops=dict(arrowstyle='-|>', color='black', lw=0, mutation_scale=20))

# Second arrow: draw dotted line, then solid arrowhead
ax.plot([2.15, 2.85], [6.85, 6.15], 'k:', linewidth=2)  # dotted line
ax.annotate('', xy=(3, 6), xytext=(2.85, 6.15), 
            arrowprops=dict(arrowstyle='-|>', color='black', lw=0, mutation_scale=20))


ax.plot([6.5, 6.5], [6.65, 7.3], 'k-', linewidth=2)  # dotted line
ax.annotate('', xy=(6.5, 7.5), xytext=(6.5, 6.5), 
            arrowprops=dict(arrowstyle='-|>', color='black', lw=0, mutation_scale=20))

ax.plot([3.5, 3.5], [6.75, 7.4], 'k-', linewidth=2)  # dotted line
ax.annotate('', xy=(3.5, 6.5), xytext=(3.5, 7.5), 
            arrowprops=dict(arrowstyle='-|>', color='black', lw=0, mutation_scale=20))

ax.plot([5, 5], [5.35, 4.65], 'k-', linewidth=2)  # dotted line
ax.annotate('', xy=(5, 4.5), xytext=(5, 5.5), 
            arrowprops=dict(arrowstyle='-|>', color='black', lw=0, mutation_scale=20))

ax.text(5, 7, 'CO$_2$', ha='center', va='center', rotation=0, fontsize=12)
ax.text(6.5, 5, 'CO$_2$', ha='center', va='center', rotation=0, fontsize=12)

ax.annotate('', xy=(5.55, 7), xytext=(6.5, 6.65),
            arrowprops=dict(arrowstyle='-|>', color='black', lw=.75, 
                          connectionstyle='angle3,angleA=75,angleB=0', mutation_scale=10))

# Add curved arrow from middle of rectangular arrow to CO2 label
ax.annotate('', xy=(4.55, 7), xytext=(3.5, 7.25),
            arrowprops=dict(arrowstyle='-|>', color='black', lw=1, 
                          connectionstyle='angle3,angleA=75,angleB=0', mutation_scale=10))


ax.annotate('', xy=(7, 5), xytext=(8, 5.5),
            arrowprops=dict(arrowstyle='-|>', color='black', lw=.75, 
                          connectionstyle='angle3,angleA=90,angleB=0', mutation_scale=10))


ax.annotate('', xy=(6, 5), xytext=(5, 5.25),
            arrowprops=dict(arrowstyle='-|>', color='black', lw=.75, 
                          connectionstyle='angle3,angleA=90,angleB=0', mutation_scale=10))



# Rectangular CO2 arrow from passive to fast pool with filled arrowheads on both ends
# Draw the rectangular path as a line
ax.plot([7.4, 8, 8, 7.4], [4, 4, 8, 8], 'k-', linewidth=4)

# Add filled arrowheads at both ends
ax.annotate('', xy=(7, 4), xytext=(7.15, 4), 
            arrowprops=dict(arrowstyle='-|>', color='black', lw=0, mutation_scale=40))
ax.annotate('', xy=(7, 8), xytext=(7.15, 8), 
            arrowprops=dict(arrowstyle='-|>', color='black', lw=0, mutation_scale=40))

# Add curved arrow from middle of rectangular arrow to CO2 label
ax.annotate('', xy=(8.5, 6), xytext=(8, 6),
            arrowprops=dict(arrowstyle='-|>', color='black', lw=.75, 
                          connectionstyle='angle3,angleA=90,angleB=25', mutation_scale=10))

# Add the CO2 label for the rectangular arrow
ax.text(8.5, 6, 'CO$_2$', ha='left', va='center', rotation=0, fontsize=12)

# Save and show the plot
# plt.savefig("soil_carbon_model_rectangular.png")


plt.sca(axs['B'])
# stem plot of J_t
# add label (a)
ax.text(10, 10, 'B', ha='center', va='center', fontsize=14)


for i, (J, color) in enumerate(zip(J_t[:5], color_order[:5])):
    plt.stem(i, J, color)
    
plt.xlabel('time')
plt.xticks(np.arange(0, 8, 5))
plt.xlim(-1, 8)
plt.ylim(0, 14)
plt.text(5.5, 5, '...', fontsize=12, ha='center', va='center')
plt.ylabel(r'carbon inputs $J(t)$')
plt.title('inputs over time')

plt.sca(axs['C'])
plt.stem(0, 1, color_order[0])
ages = np.arange(100)
plt.plot(ages, decay(1, ages), color=color_order[0])
plt.xlabel('age')
#plt.xticks(np.arange(len(J_t)))
plt.xlim(-1.5, 20)
plt.ylim(0, 1.4)
plt.ylabel(r'remaining carbon $s(\tau)$')
plt.title('decay with age')

plt.sca(axs['D'])

# Only have 10 colors -- plot the first 10
for i, (J, color) in enumerate(zip(J_t[:10], color_order[:10])):
    plt.stem(i, J, color)

g_ts = np.zeros((len(J_t), len(ages)+len(J_t) + 10))
for i in range(len(J_t)):
    decay_i = decay(J_t[i], ages)
    g_ts[i, i:i+len(decay_i)] = decay_i

    # plot the first 10 only 
    if i >= 10:
        continue
    plt.plot(ages + i, decay_i, color=color_order[i])
plt.xlabel('time')
plt.ylabel(r'$J(t-\tau)\cdot s(\tau)$')
plt.text(15.5, 5, '...', fontsize=12, ha='center', va='center')
plt.xlim(-1, 20)
plt.ylim(0, 14)
plt.title('inputs with different ages decay')

# plot the stocks over time, which is the sum of 
# what remains from prior inputs
plt.sca(axs['E'])

nts = g_ts.shape[1]
njs = g_ts.shape[0]
bottom = np.zeros(nts)
ts = np.arange(nts)
for i in range(njs):
    color = color_order[i % len(color_order)]
    top = bottom + g_ts[i, :]
    plt.fill_between(ts, bottom, top, color=color, alpha=0.7, 
                     edgecolor='k', lw=0.2)
    bottom = top

G_t = np.sum(g_ts, axis=0)
plt.plot(ts, G_t, color='black', lw=2)
plt.xlabel('time')
plt.xlim(0, 50)
plt.ylim(0, 83)
plt.ylabel(r'total carbon stocks $G(t)$')
plt.title(r'stocks = sum of residual inputs')

plt.tight_layout()

plt.savefig('figures/fig1.png', dpi=600)


