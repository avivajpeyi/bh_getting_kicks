import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv

BILBY_BLUE_COLOR = '#0072C1'
VIOLET_COLOR = "#8E44AD"

RADIUS = 1
N_VEC = 100
MIN_COS_THETA = -1

CORNER_KWARGS = dict(
    smooth=0.9,
    label_kwargs=dict(fontsize=30),
    title_kwargs=dict(fontsize=16),
    color='#0072C1',
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


def get_isotropic_vector(min_cos_theta=-1):
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    :return:
    """
    phi = np.random.uniform(0, np.pi * 2)
    costheta = np.random.uniform(min_cos_theta, 1)

    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return [x, y, z]


def compute_vectors(mesh):
    origin = 0
    vectors = mesh.points - origin
    vectors = normalise_vectors(vectors)
    return vectors


def normalise_vectors(vectors):
    return vectors / np.linalg.norm(vectors, axis=1)[:, None]


def plot_spherical_vectors():
    p = pv.Plotter()
    ar_kwgs = dict(scale=RADIUS * 2, shaft_radius=0.01, tip_radius=0.05, tip_length=0.1)
    p.add_mesh(pv.Sphere(radius=RADIUS))
    p.add_mesh(pv.Arrow(direction=[1, 0, 0], **ar_kwgs), color="blue")  # x
    p.add_mesh(pv.Arrow(direction=[0, 1, 0], **ar_kwgs), color="red")  # y
    p.add_mesh(pv.Arrow(direction=[0, 0, 1], **ar_kwgs), color="green")  # Z

    def create_mesh(min_cos_theta):
        vectors = [get_isotropic_vector(min_cos_theta) for _ in range(N_VEC)]
        pt_cloud = pv.PolyData(vectors)
        vectors = compute_vectors(pt_cloud)
        pt_cloud['vectors'] = vectors
        arrows = pt_cloud.glyph(orient='vectors', scale=False, factor=0.3, )
        p.remove_actor('pts')
        p.remove_actor('arrows')
        p.add_mesh(pt_cloud, color='maroon', point_size=10,
                   render_points_as_spheres=True, name='pts')
        p.add_mesh(arrows, color='lightblue', name='arrows')
        plot_corner_of_spins(min_cos_theta)

    p.add_slider_widget(create_mesh, rng=[-1, 1], value=1, title='Min Cos(tilt)')
    p.show()


def get_zenith_angle(z):
    """Angle from z to vector [0, pi)"""
    return np.arccos(z)


def get_azimuth_angle(x, y):
    """angle bw north vector and projected vector on the horizontal plane [0, 2pi]"""
    azimuth = np.arctan2(y, x)  # [-pi, pi)
    if azimuth < 0.0:
        azimuth += 2 * np.pi
    return azimuth


def convert_vectors_to_bbh_param(min_cos_theta=MIN_COS_THETA):
    """Generate BBH spin vectors and convert to LIGO BBH params
    cos_tilt_i:
        Cosine of the zenith angle between the s and j [-1,1]
    Phi_12:
        diff bw azimuthal angles of the s1+s2 projections on orbital plane [0, 2pi]
    phi_jl:
        diff bw L and J azimuthal angles [0, 2pi]
    """
    n_vec = 1000
    # s1 is pointing along Z
    s1 = [[0, 0, 1] for _ in range(n_vec)]
    # s2 is slightly off Z
    s2 = normalise_vectors([get_isotropic_vector(min_cos_theta) for _ in range(n_vec)])

    # J isotropically distributed
    j = normalise_vectors([get_isotropic_vector() for _ in range(n_vec)])

    # L is s1 + s2
    l = normalise_vectors(s1 + s2)

    return pd.DataFrame(dict(
        cos_tilt_1=np.cos([get_zenith_angle(v[2]) for v in s1]),
        cos_tilt_2=np.cos([get_zenith_angle(v[2]) for v in s2]),
        phi_12=get_difference_in_azimuths(s1, s2),
        phi_jl=get_difference_in_azimuths(j, l),
    ))


def get_difference_in_azimuths(v1, v2):
    v1_azimuth = [get_azimuth_angle(v[0], v[1]) for v in v1]
    v2_azimuth = [get_azimuth_angle(v[0], v[1]) for v in v2]
    angles = [j - i for i, j in zip(v1_azimuth, v2_azimuth)]
    return np.mod(angles, 2 * np.pi)


def plot_one_bh_param(s1, s2, l, j):
    my_vectors = [s1, s2, l, j]

    # Show the result
    p = pv.Plotter()
    p.add_mesh(pv.Sphere(radius=RADIUS))
    ar_kwgs = dict(scale=RADIUS * 2, shaft_radius=0.01, tip_radius=0.05, tip_length=0.1)
    p.add_mesh(pv.Arrow(direction=[1, 0, 0], **ar_kwgs), color="blue")  # x
    p.add_mesh(pv.Arrow(direction=[0, 1, 0], **ar_kwgs), color="red")  # y
    p.add_mesh(pv.Arrow(direction=[0, 0, 1], **ar_kwgs), color="green")  # Z
    for v in my_vectors:
        pt_cloud = pv.PolyData(v)
        vectors = compute_vectors(pt_cloud)
        pt_cloud['vectors'] = vectors
        p.add_mesh(pt_cloud, color='maroon', point_size=10,
                   render_points_as_spheres=True)
        arrows = pt_cloud.glyph(orient='vectors', scale=False, factor=0.3)
        p.add_mesh(arrows, color='maroon')
    p.show()


def plot_corner_of_spins(min_cos_theta):
    bbh_vectors = convert_vectors_to_bbh_param(min_cos_theta)
    kwargs = dict(
        smooth=0.9,
        label_kwargs=dict(fontsize=30),
        title_kwargs=dict(fontsize=16),
        color='#0072C1',
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
    corner.corner(bbh_vectors, **kwargs,
                  range=[(-1, 1), (-1, 1), (0, 2 * np.pi), (0, 2 * np.pi)])
    plt.savefig(f"spin_params_{min_cos_theta:.2f}.png")


def main():
    plot_spherical_vectors()


if __name__ == '__main__':
    # main()
    plot_one_bh_param(s1=[0,0,1], s2=[0.5, 0, 0.5], l=[0.33,0.33,0.33], j=[0,0,1])
