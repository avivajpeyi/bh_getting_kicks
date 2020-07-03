import numpy as np
COLOR = '#0072C1'

CORNER_KWARGS = dict(
    smooth=0.9,
    label_kwargs=dict(fontsize=16),
    title_kwargs=dict(fontsize=16),
    color='#0072C1',
    truth_color='tab:orange',
    quantiles=[0.16, 0.84],
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=True,
    show_titles=True,
    max_n_ticks=3,
)


plt.hist(df.precessing, density=True, alpha=0.5,  bins=50, label='Precessing Spin')
plt.hist(df.aligned, density=True, alpha=0.5, bins=5, label='Aligned Spin')
plt.xlabel("Kick Mag")
plt.ylabel("Density")
plt.legend()
plt.savefig("kicks_one_plot.png")
plt.close()


from mpl_toolkits import mplot3d