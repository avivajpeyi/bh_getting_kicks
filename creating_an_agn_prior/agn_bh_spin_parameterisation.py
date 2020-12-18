import os
import shutil

import corner
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import scipy.stats
from matplotlib import rc
from tqdm import tqdm

rc('text', usetex=True)

N_VEC = "Num BBH"
COS_theta_12 = "cos(theta_12)"
COS_theta_1L = "cos(theta_1L)"

BILBY_BLUE_COLOR = '#0072C1'
VIOLET_COLOR = "#8E44AD"

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


def rotate_vector_along_z(v1, theta):
    """
    |cos θ   −sin θ   0| |x|   |x cos θ − y sin θ|   |x'|
    |sin θ    cos θ   0| |y| = |x sin θ + y cos θ| = |y'|
    |  0       0      1| |z|   |        z        |   |z'|
    """
    x, y, z = v1[0], v1[1], v1[2]
    return [
        x * np.cos(theta) - y * np.sin(theta),
        x * np.sin(theta) + y * np.cos(theta),
        z
    ]


def rotate_vector_along_y(v1, theta):
    """
    | cos θ    0   sin θ| |x|   | x cos θ + z sin θ|   |x'|
    |   0      1       0| |y| = |         y        | = |y'|
    |−sin θ    0   cos θ| |z|   |−x sin θ + z cos θ|   |z'|
    """
    x, y, z = v1[0], v1[1], v1[2]
    return [
        x * np.cos(theta) + z * np.sin(theta),
        y,
        - x * np.sin(theta) + z * np.cos(theta),
    ]


def get_isotropic_vector(std=1):
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    :return:
    """
    theta = np.random.uniform(0, np.pi * 2)
    # truncated normal distribution --> peaks at costheta = 1
    # hyperparam --> sigma
    # costheta = np.random.uniform(std, 1)

    mean = 1
    clip_a, clip_b = -1, 1

    if std == 0:
        std = 0.00001
    a, b = (clip_a - mean) / std, (clip_b - mean) / std
    costheta = scipy.stats.truncnorm.rvs(
        a=a,
        b=b,
        loc=mean,
        scale=std,
        size=1)[0]
    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(theta)
    y = np.sin(theta) * np.sin(theta)
    z = np.cos(theta)
    return [x, y, z]


def rotate_v2_to_v1(v1, v2):
    azimuth = get_azimuth_angle(v1[0], v1[1])
    zenith = get_zenith_angle(v1[2])
    v2 = rotate_vector_along_y(v2, zenith)
    v2 = rotate_vector_along_z(v2, azimuth)
    return v2


def compute_vectors(mesh):
    origin = 0
    vectors = mesh.points - origin
    vectors = normalise_vectors(vectors)
    return vectors


def normalise_vectors(vectors):
    return vectors / np.linalg.norm(vectors, axis=1)[:, None]


class SphereAngleAnimation():
    def __init__(self):
        # default parameters
        self.kwargs = {
            'radius': 1,
            N_VEC: 100,
            COS_theta_1L: 1,
            COS_theta_12: 1,
        }
        self.s1_color = "lightblue"
        self.s2_color = "lightgreen"
        self.plotter = self.init_plotter()
        self.add_sliders()
        self.plotter.show("AGN BBH spins")
        self.add_vectors()

    def __call__(self, param, value):
        self.kwargs[param] = value
        self.update()

    def add_sliders(self):
        LEFT = dict(pointa=(.025, .1), pointb=(.31, .1), )
        MIDDLE = dict(pointa=(.35, .1), pointb=(.64, .1))
        RIGHT = dict(pointa=(.67, .1), pointb=(.98, .1), )

        self.plotter.add_slider_widget(
            callback=lambda value: self(COS_theta_1L, value),
            rng=[0, 1],
            value=1,
            title=f"min {COS_theta_1L}",
            style='modern',
            **LEFT
        )
        self.plotter.add_slider_widget(
            callback=lambda value: self(COS_theta_12, value),
            rng=[0, 1],
            value=1,
            title=f"min {COS_theta_12}",
            style='modern',
            **MIDDLE
        )
        self.plotter.add_slider_widget(
            callback=lambda value: self(N_VEC, int(value)),
            rng=[1, 1000],
            value=100,
            title=N_VEC,
            style='modern',
            **RIGHT
        )

    def init_plotter(self):
        p = pv.Plotter()
        p.add_mesh(pv.Sphere(radius=self.kwargs['radius']))
        ar_kwgs = dict(
            scale=self.kwargs['radius'] * 2,
            shaft_radius=0.01,
            tip_radius=0.05,
            tip_length=0.1
        )
        p.add_mesh(pv.Arrow(direction=[1, 0, 0], **ar_kwgs), color="blue")  # x
        p.add_mesh(pv.Arrow(direction=[0, 1, 0], **ar_kwgs), color="red")  # y
        p.add_mesh(pv.Arrow(direction=[0, 0, 1], **ar_kwgs), color="green")  # Z
        p.add_legend(labels=[
            ["L", "green"],
            ["S1", self.s1_color],
            ["S2", self.s2_color]
        ])
        return p

    def add_vectors(self):
        s1_vectors = [get_isotropic_vector(self.kwargs[COS_theta_1L]) for _ in
                      range(self.kwargs[N_VEC])]
        s2_vectors = [get_isotropic_vector(self.kwargs[COS_theta_12]) for _ in
                      range(self.kwargs[N_VEC])]
        s2_vectors = [rotate_v2_to_v1(s1, s2) for s1, s2 in zip(s1_vectors, s2_vectors)]

        self.add_vector_list(s1_vectors, name="s1", color=self.s1_color)
        self.add_vector_list(s2_vectors, name="s2", color=self.s2_color)

    def add_vector_list(self, vectors, name, color):
        self.plotter.remove_actor(f'{name}_pts')
        self.plotter.remove_actor(f'{name}_arrows')
        pt_cloud = pv.PolyData(vectors)
        vectors = compute_vectors(pt_cloud)
        pt_cloud['vectors'] = vectors
        arrows = pt_cloud.glyph(orient='vectors', scale=False, factor=0.3, )
        self.plotter.add_mesh(pt_cloud, color=color, point_size=10,
                              render_points_as_spheres=True, name=f'{name}_pts')
        self.plotter.add_mesh(arrows, color=color, name=f'{name}_arrows')

    def update(self):
        self.add_vectors()


def get_zenith_angle(z):
    """Angle from z to vector [0, pi)"""
    return np.arccos(z)


def get_azimuth_angle(x, y):
    """angle bw north vector and projected vector on the horizontal plane [0, 2pi]"""
    azimuth = np.arctan2(y, x)  # [-pi, pi)
    if azimuth < 0.0:
        azimuth += 2 * np.pi
    return azimuth


def get_chi_eff(s1, s2, q=1):
    s1z, s2z = s1[2], s2[2]
    return (s1z * s2z) * (q / (1 + q))


def get_chi_p(s1, s2, q=1):
    chi1p = np.sqrt(s1[0] ** 2 + s1[1] ** 2)
    chi2p = np.sqrt(s2[0] ** 2 + s2[1] ** 2)
    qfactor = q * ((4 * q) + 3) / (4 + (3 * q))
    return np.maximum(
        chi1p,
        chi2p * qfactor
    )


def convert_vectors_to_bbh_param(cos_theta1L_std, cos_theta12_std):
    """Generate BBH spin vectors and convert to LIGO BBH params
    cos_tilt_i:
        Cosine of the zenith angle between the s and j [-1,1]
    theta_12:
        diff bw azimuthal angles of the s1hat+s2 projections on orbital plane [0, 2pi]
    theta_jl:
        diff bw L and J azimuthal angles [0, 2pi]
    """
    n = 1000
    lhat = normalise_vectors([[0, 0, 1] for _ in range(n)])
    s1hat = normalise_vectors([get_isotropic_vector(cos_theta1L_std) for _ in range(n)])
    s2hat = normalise_vectors([get_isotropic_vector(cos_theta12_std) for _ in range(n)])
    s2hat = normalise_vectors(
        [rotate_v2_to_v1(s1v, s2v) for s1v, s2v in zip(s1hat, s2hat)])

    return pd.DataFrame(dict(
        cos_tilt_1=np.cos([get_zenith_angle(v[2]) for v in s1hat]),
        cos_tilt_2=np.cos([get_zenith_angle(v[2]) for v in s2hat]),
        chi_eff=[get_chi_eff(s1, s2) for s1, s2 in zip(s1hat, s2hat)],
        chi_p=[get_chi_p(s1, s2) for s1, s2 in zip(s1hat, s2hat)],
        cos_theta_12=[np.cos(get_angle_bw_vectors(s1, s2)) for s1, s2 in zip(s1hat, s2hat)],
        cos_theta_1L=[np.cos(get_angle_bw_vectors(s1, l)) for s1, l in zip(s1hat, lhat)],
    ))


def get_angle_bw_vectors(v1, v2):
    unit_vector1 = v1 / np.linalg.norm(v1)
    unit_vector2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector1, unit_vector2)
    return np.arccos(dot_product)


def plot_corner_of_spins(cos_theta1L_std, cos_theta12_std, save=True):
    bbh_vectors = convert_vectors_to_bbh_param(cos_theta1L_std=cos_theta1L_std,
                                               cos_theta12_std=cos_theta12_std)
    labels = [PARAMS[p]['l'] for p in bbh_vectors.columns.values]
    range = [PARAMS[p]['r'] for p in bbh_vectors.columns.values]
    corner.corner(bbh_vectors, **CORNER_KWARGS, labels=labels, range=range)
    if save:
        plt.savefig(f"spins_theta1L{cos_theta1L_std:.2f}_theta12{cos_theta12_std:.2f}.png")


def plot_overlaid_corners(cos_theta1L_std_vals, cos_theta12_std_vals, pltdir):
    base = convert_vectors_to_bbh_param(cos_theta1L_std=1, cos_theta12_std=1)
    labels = [PARAMS[p]['l'] for p in base.columns.values]
    range = [PARAMS[p]['r'] for p in base.columns.values]
    kwargs = dict(**CORNER_KWARGS, labels=labels, range=range)

    if os.path.isdir(pltdir):
        shutil.rmtree(pltdir)
    os.makedirs(pltdir, exist_ok=False)

    i = 0
    for min_cos_theta1L, min_cos_theta12 in tqdm(
            zip(cos_theta1L_std_vals, cos_theta12_std_vals),
            total=len(cos_theta1L_std_vals), desc="Hyper-Param settings"
    ):
        f = f"{pltdir}/{i:02}_p12{min_cos_theta12:.1f}_p1L{min_cos_theta1L:.1f}.png"
        compare = convert_vectors_to_bbh_param(
            cos_theta1L_std=min_cos_theta1L,
            cos_theta12_std=min_cos_theta12)
        compare.to_csv(f.replace(".png", ".csv"))
        normalising_weights = np.ones(len(compare)) * len(compare) / len(compare)
        fig = corner.corner(base, **kwargs, color=BILBY_BLUE_COLOR)
        corner.corner(compare, fig=fig, weights=normalising_weights, **kwargs,
                      color=VIOLET_COLOR)

        orig_line = mlines.Line2D([], [], color=BILBY_BLUE_COLOR,
                                  label="Isotropic Spins")
        weighted_line = mlines.Line2D(
            [], [], color=VIOLET_COLOR,
            label=f"Adjusted spins m(cos(p12))={min_cos_theta12:.1f}, m(cos(p1L))={min_cos_theta1L:.1f}"
        )
        plt.legend(handles=[orig_line, weighted_line], fontsize=25,
                   frameon=False,
                   bbox_to_anchor=(1, len(labels)), loc="upper right")
        plt.savefig(f)
        plt.close()
        i += 1


if __name__ == '__main__':
    # r = SphereAngleAnimation()

    cos_theta12_std_vals = list(np.arange(0, 2.1, 0.1))
    cos_theta1L_std_vals = [1 for i in range(len(cos_theta12_std_vals))]

    plot_overlaid_corners(cos_theta1L_std_vals=cos_theta12_std_vals,
                          cos_theta12_std_vals=cos_theta1L_std_vals, pltdir="test")
