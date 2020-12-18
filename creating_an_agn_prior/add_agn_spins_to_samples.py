import glob
import os

import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

import matplotlib.lines as mlines

VIOLET_COLOR = "#8E44AD"
BILBY_BLUE_COLOR = '#0072C1'

PARAMS = dict(
    chi_eff=dict(l=r"$\chi_{eff}$", r=(-1, 1)),
    chi_p=dict(l=r"$\chi_{p}$", r=(-1, 1)),
    cos_tilt_1=dict(l=r"$\cos(t1)$", r=(-1, 1)),
    cos_tilt_2=dict(l=r"$\cos(t2)$", r=(-1, 1)),
    cos_theta_12=dict(l=r"$\cos \theta_{12}$", r=(-1, 1)),
    cos_theta_1L=dict(l=r"$\cos \theta_{1L}$", r=(-1, 1)),
)

CORNER_KWARGS = dict(
    smooth=0.9,
    label_kwargs=dict(fontsize=30),
    title_kwargs=dict(fontsize=16),
    truth_color='tab:orange',
    quantiles=[0.16, 0.84],
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=True,
    max_n_ticks=3,
    verbose=False,
    use_math_text=True,
)


def add_agn_samples_to_df(df):
    df['s1x'], df['s1y'], df['s1z'] = df['spin_1x'], df['spin_1y'], df['spin_1z']
    df['s2x'], df['s2y'], df['s2z'] = df['spin_2x'], df['spin_2y'], df['spin_2z']
    df['s1_dot_s2'] = (df['s1x'] * df['s2x']) + (df['s1y'] * df['s2y']) + (
                df['s1z'] * df['s2z'])
    df['s1_mag'] = np.sqrt(df['s1x'] * df['s1x']) + (df['s1y'] * df['s1y']) + (
                df['s1z'] * df['s1z'])
    df['s2_mag'] = np.sqrt(df['s2x'] * df['s2x']) + (df['s2y'] * df['s2y']) + (
                df['s2z'] * df['s2z'])
    df['cos_theta_12'] = df['s1_dot_s2'] / (df['s1_mag'] * df['s2_mag'])
    # Lhat = [0, 0, 1]
    df['cos_theta_1L'] = df['s1z'] / (df['s1_mag'])
    return df


def calculate_weight(df, sigma):
    mean = 1
    clip_a, clip_b = -1, 1
    a, b = (clip_a - mean) / sigma, (clip_b - mean) / sigma
    costheta_prior = scipy.stats.truncnorm(a=a, b=b, loc=mean, scale=sigma)
    df['weight'] = np.abs(np.exp(costheta_prior.logpdf(df['cos_theta_1L'])))
    df['weight'] = df['weight'] * np.abs(np.exp(costheta_prior.logpdf(df['cos_theta_12'])))
    return df

def process_res(r):
    params = list(PARAMS.keys())
    labels = [PARAMS[p]['l'] for p in params]
    range = [PARAMS[p]['r'] for p in params]

    label = os.path.basename(r).split("_")[0]
    df = pd.read_csv(r, " ")
    df = add_agn_samples_to_df(df)
    df = calculate_weight(df, sigma=0.5)
    kwargs = CORNER_KWARGS.copy()
    kwargs.update(labels=labels, range=range)

    fig = corner.corner(df[params], color=BILBY_BLUE_COLOR, **kwargs)


    #
    # dfnew = df.copy()
    # dfnew = dfnew.sample(
    #     frac=0.1,
    #     weights='weight',
    #     replace=True
    # )
    # normalising_weights_s2 = np.ones(len(dfnew)) * len(df) / len(dfnew)
    # fig = corner.corner(dfnew[params], color=VIOLET_COLOR, weights=normalising_weights_s2,  fig=fig, **kwargs)
    #
    # orig_line = mlines.Line2D([], [], color=BILBY_BLUE_COLOR, label="Original Posterior")
    # weighted_line = mlines.Line2D([], [], color=VIOLET_COLOR,label=r"Reweighted Posterior" )
    # plt.legend(handles=[orig_line, weighted_line], fontsize=25,
    #            frameon=False,
    #            bbox_to_anchor=(1, len(PARAMS)), loc="upper right")

    fig.suptitle(label, fontsize=24)
    fig.savefig(f"{label}.png")
    plt.close(fig)


def main():
    res = glob.glob(
        "../datafiles/samples/downsampled_posterior_samples_v1.0.0/*_samples.dat")
    for r in res:
        try:
            process_res(r)
            print(f"Processed {r}")
        except Exception:
            print(f"Skipping {r}")


if __name__ == '__main__':
    main()
    process_res("../datafiles/samples/downsampled_posterior_samples_v1.0.0/GW150914_downsampled_posterior_samples.dat")