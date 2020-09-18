import numpy as np

BILBY_BLUE_COLOR = '#0072C1'
VIOLET_COLOR = "#8E44AD"

CORNER_KWARGS = dict(
    smooth=0.9,
    label_kwargs=dict(fontsize=30),
    title_kwargs=dict(fontsize=16),
    color=BILBY_BLUE_COLOR,
    truth_color='tab:orange',
    quantiles=[0.16, 0.84],
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=True,
    max_n_ticks=3,
    verbose=True,
    use_math_text=True
)

if __name__ == '__main__':
    import corner
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib import rcParams

    rcParams["font.size"] = 20
    rcParams["font.family"] = "serif"
    rcParams["font.sans-serif"] = ["Computer Modern Sans"]
    rcParams["text.usetex"] = False
    rcParams['axes.labelsize'] = 30
    rcParams['axes.titlesize'] = 30
    rcParams['axes.labelpad'] = 20
    samples = pd.DataFrame(dict(
        m1=np.random.normal(loc=35, scale=15, size=1000),
        m2=np.random.normal(loc=40, scale=30, size=1000)
    ))
    truths = [35+20, 40-5]
    corner.corner(samples, **CORNER_KWARGS, truths=truths)
    plt.savefig("bayesian.png")
    CORNER_KWARGS.update(dict(
        fill_contours=False,
        quantiles=None,
        alpha=0,
        color='white'
    ))
    corner.corner(samples, **CORNER_KWARGS, truths=truths)
    plt.savefig("freq.png")

