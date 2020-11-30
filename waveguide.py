import collections

import numpy as np
import scipy.interpolate

from port import Port
from gdshelpers.helpers import find_line_intersection, normalize_phase
from gdshelpers.helpers.bezier import CubicBezierCurve
from gdshelpers.helpers.positive_resist import convert_to_positive_resist


class Waveguide:
    def __init__(self, origin, angle):
        self._current_port = Port(origin, angle)
        self._segments = list()

    @classmethod
    def make_at_port(cls, port, **kargs):
        port_param = port.copy()
        port_param.set_port_properties(**kargs)
        return cls(**port_param.get_parameters())

    @property
    def x(self):
        return self._current_port.origin[0]

    @property
    def y(self):
        return self._current_port.origin[1]

    @property
    def z(self):
        return self._current_port.origin[2]

    @property
    def origin(self):
        return self._current_port.origin

    @property
    def angle(self):
        return self._current_port.angle

    @property
    def current_port(self):
        return self._current_port.copy()

    # Add alias for current port
    port = current_port

    @property
    def in_port(self):
        return self._in_port.copy()

    @property
    def length(self):
        return sum((length for port, obj, length in self._segments))

    @property
    def length_last_segment(self):
        if not len(self._segments):
            return 0
        return self._segments[-1][3]

    def get_segments(self):
        """
        Returns the list of tuples, containing their ports and shapely objects.
        """
        return [(port.copy(), obj, length) for port, obj, length in self._segments]

    def add_straight_segment(self, length, **kwargs):
        self._current_port.set_port_properties(**kwargs)

        if not np.isclose(length, 0):
            assert length >= 0, 'Length of straight segment must not be negative'

            self.add_parameterized_path(path=lambda t: [t * length, 0, 0],
                                        path_derivative=lambda t: [1, 0, 0],
                                        sample_points=2,
                                        sample_distance=0)
        return self

    def add_arc(self, final_angle, radius, n_points=128, shortest=True, **kwargs):
        delta = final_angle - self.angle
        if not np.isclose(normalize_phase(delta), 0):
            if shortest:
                delta = normalize_phase(delta)
            self.add_bend(delta, radius, n_points, **kwargs)
        return self

    def add_bend(self, angle, radius, n_points=128, **kwargs):
        # If number of points is None, default to 128
        n_points = n_points if n_points else 128

        self._current_port.set_port_properties(**kwargs)
        sample_points = max(int(abs(angle) / (np.pi / 2) * n_points), 2)

        angle = normalize_phase(angle, zero_to_two_pi=True) - \
            (0 if angle > 0 else 2 * np.pi)

        self.add_parameterized_path(
            path=lambda t: [
                radius * np.sin(abs(angle) * t), np.sign(angle) * -radius *
                (np.cos(angle * t) - 1), np.zeros(sample_points)],
            path_function_supports_numpy=True,
            path_derivative=lambda t: [radius * np.cos(abs(angle) * t) * abs(angle),
                                       np.sign(angle) * radius *
                                       (np.sin(angle * t) * angle),
                                       np.zeros(sample_points)],
            sample_points=sample_points,
            sample_distance=0)

        return self

    def add_s_bend(self, height, radius, direction, n_points=128, **kwargs):
        """
        Add an S-Bend to the waveguide.
        A S-Bend made of arcs is unique given the height of the overall bend.

        :param height: height of the whole S-Bend
        :param radius: radius of the S-Bend
        :param direction: +1 toward the top of the chip, -1 toward the bottom
        :param n_points: number of points of the arc.
        """

        # compute angle
        a = np.arccos(1-(height/2)/radius)

        if (direction == +1):
            self.add_bend(-a, radius, n_points, **kwargs)
            self.add_bend(a, radius, n_points, **kwargs)
        else:
            self.add_bend(a, radius, n_points, **kwargs)
            self.add_bend(-a, radius, n_points, **kwargs)

        return self

    def add_parameterized_path(self, path, sample_distance=0.50, sample_points=100, path_derivative=None,
                               path_function_supports_numpy=False):
        """
        Generate a parameterized path.

        The path coordinate system is the origin and rotation of the current path. So if you want to continue your path
        start at (0, 0) in y-direction.

        Note, that path is either a list of (x,y) coordinate tuples or a callable function which takes one float
        parameter between 0 and 1. If you use a parameterized function, its first derivative must be continuous.
        When using a list of coordinates, these points will be connected by straight lines. They must be sufficiently
        close together to simulate a first derivative continuous path.

        This function will try to space the final points of the curve equidistantly. To achieve this, it will first
        sample the function and find its first derivative. Afterwards it can calculate the cumulative sum of the length
        of the first derivative. This allows to sample the function nearly equidistantly in a second step. This
        approach might be wasteful for paths like (x**2, y). You can suppress resampling for length by passing zero or
        none as sample_distance parameter.

        Note, that your final direction of the path might not be what you expected. This is caused by the numerical
        procedure which generates numerical errors when calculating the first derivative. You can either append another
        arc to the waveguide to get to you a correct angle or you can also supply a function which is the algebraic
        first derivative. The returned vector is not required to be normed.

        By default, for each parameter point t, the parameterized functions are call. You will notice that this is
        rather slow. To achieve the best performance, write your functions in such a way, that they can handle a
        numpy array as parameter *t*. Once the *path_function_supports_numpy* option is set to True, the function will
        be called only once, speeding up the calculation considerable.

        :param path:
        :param sample_distance:
        :param sample_points:
        :param path_derivative:
        :param path_function_supports_numpy:
        """

        if callable(path):
            presample_t = np.linspace(0, 1, sample_points)

            if path_function_supports_numpy:
                presample_coordinates = np.array(path(presample_t)).T
            else:
                presample_coordinates = np.array(
                    [path(x) for x in presample_t])

            if sample_distance:
                # # Calculate the derivative
                # if path_derivative:
                #     assert callable(path_derivative), 'The derivative of the path function must be callable'
                #     presample_coordinates_d1 = np.array([path_derivative(x) for x in presample_t[:-1]])
                # else:
                #     presample_coordinates_d1 = np.diff(presample_coordinates, axis=0)
                presample_coordinates_d1 = np.diff(
                    presample_coordinates, axis=0)
                presample_coordinates_d1_norm = np.linalg.norm(
                    presample_coordinates_d1, axis=1)
                presample_coordinates_d1__cum_norm = np.insert(
                    np.cumsum(presample_coordinates_d1_norm), 0, 0)

                lengths = np.linspace(presample_coordinates_d1__cum_norm[0],
                                      presample_coordinates_d1__cum_norm[-1],
                                      int(presample_coordinates_d1__cum_norm[-1] / sample_distance))

                # First get the spline representation. This is needed since we manipulate these directly for roots
                # finding.
                spline_rep = scipy.interpolate.splrep(
                    presample_t, presample_coordinates_d1__cum_norm, s=0)

                def find_y(y):
                    interp_result = scipy.interpolate.sproot(
                        (spline_rep[0], spline_rep[1] - y, spline_rep[2]), mest=1)
                    return interp_result[0] if len(interp_result) else None

                # We need a small hack here and exclude lengths[0]==0 since it finds no root there
                sample_t = np.array([0, ] + [find_y(length)
                                             for length in lengths[1:-1]] + [1, ])

                if path_function_supports_numpy:
                    sample_coordinates = np.array(path(sample_t)).T
                else:
                    sample_coordinates = np.array([path(x) for x in sample_t])
            else:
                sample_coordinates = presample_coordinates
                sample_t = presample_t
        else:
            # If we do not have a sample function, we need to "invent a sampling parameter"
            sample_coordinates = np.array(path)
            sample_t = np.linspace(0, 1, sample_coordinates.shape[0])

        rotation_matrix = np.array(((np.cos(self._current_port.angle),
                                     -np.sin(self._current_port.angle), 0),
                                    (np.sin(self._current_port.angle),
                                     np.cos(self._current_port.angle), 0),
                                    (0, 0, 1)))
        sample_coordinates = self._current_port.origin + \
            np.einsum('ij,kj->ki', rotation_matrix, sample_coordinates)

        # Calculate the derivative
        if callable(path_derivative):
            if path_function_supports_numpy:
                sample_coordinates_d1 = np.array(path_derivative(sample_t)).T
            else:
                sample_coordinates_d1 = np.array(
                    [path_derivative(x) for x in sample_t])
            sample_coordinates_d1 = np.einsum(
                'ij,kj->ki', rotation_matrix, sample_coordinates_d1)
        else:
            if path_derivative is None:
                sample_coordinates_d1 = np.vstack(
                    (rotation_matrix[:, 0], np.diff(sample_coordinates, axis=0)))
            else:
                sample_coordinates_d1 = np.array(path_derivative)
                sample_coordinates_d1 = np.einsum(
                    'ij,kj->ki', rotation_matrix, sample_coordinates_d1)

        sample_coordinates_d1_norm = np.linalg.norm(
            sample_coordinates_d1, axis=1)
        sample_coordinates_d1_normed = sample_coordinates_d1 / \
            sample_coordinates_d1_norm[:, None]

        # Find the orthogonal vectors to the derivative
        sample_coordinates_d1_normed_ortho = np.vstack((sample_coordinates_d1_normed[:, 1],
                                                        -sample_coordinates_d1_normed[:, 0])).T

        length = np.sum(np.linalg.norm(
            np.diff(sample_coordinates, axis=0), axis=1))
        self._segments.append(
            (self._current_port.copy(), length, sample_coordinates))

        self._current_port.origin = sample_coordinates[-1]
        self._current_port.angle = np.arctan2(
            sample_coordinates_d1[-1][1], sample_coordinates_d1[-1][0])
        return self

    def add_cubic_bezier_path(self, p0, p1, p2, p3, **kwargs):
        """
        Add a cubic bezier path to the waveguide.

        Coordinates are in the "waveguide tip coordinate system", so the first
        point will probably be p0 == (0, 0, depth).
        Note that your bezier curve undergoes the same restrictions as a parameterized path. Don't self-intersect it and don't use small bend radii.

        :param p0: 3 element tuple like coordinates
        :param p1: 3 element tuple like coordinates
        :param p2: 3 element tuple like coordinates
        :param p3: 3 element tuple like coordinates
        :param kwargs: Optional keyword arguments, passed to :func:add_parameterized_path
        :return: Changed waveguide
        :rtype: Waveguide
        """

        assert len(p0) == 3
        assert len(p1) == 3
        assert len(p2) == 3
        assert len(p3) == 3

        bezier_curve = CubicBezierCurve(p0, p1, p2, p3)

        self.add_parameterized_path(
            path=bezier_curve.evaluate, path_derivative=bezier_curve.evaluate_d1, path_function_supports_numpy=True, **kwargs)
        return self

    def add_bezier_to(self, final_coordinates, final_angle, bend_strength, **kwargs):
        """
        Add a Bezier curve that connect the current point to final_coordinates.

        :param final_coordinates: final destination
        :param final_angle: angle of arriving port
        :param bend_strength: maximum curvature radius
        :param kwargs: Optional keyword arguments, passed to :func:add_parameterized_path
        :return: Changed waveguide
        :rtype: Waveguide
        """
        try:
            bs1, bs2 = float(bend_strength[0]), float(bend_strength[1])
        except (KeyError, TypeError):
            bs1 = bs2 = float(bend_strength)

        final_port = Port(final_coordinates, final_angle)
        p0 = (0, 0, self._current_port.z)
        p1 = self._current_port.longitudinal_offset(
            bs1).origin - self._current_port.origin
        p2 = final_port.longitudinal_offset(-bs2).origin - \
            self._current_port.origin
        p3 = final_coordinates - self._current_port.origin

        tmp_wg = Waveguide.make_at_port(
            self._current_port.copy().set_port_properties(angle=0))
        tmp_wg.add_cubic_bezier_path(p0, p1, p2, p3, **kwargs)

        self._segments.append(
            (self._current_port.copy(), tmp_wg.length, tmp_wg.center_coordinates))
        self._current_port = tmp_wg.current_port

        return self

    def add_bezier_to_port(self, port, bend_strength, **kwargs):

        self.add_bezier_to(
            port.origin, port.inverted_direction.angle, bend_strength, **kwargs)
        return self


def _example():

    prof = 0.035
    indr = 1.5/1.33
    z_coord = prof/indr

    port = Port([0, 0, z_coord], 0)

    path = Waveguide.make_at_port(port)
    path.add_straight_segment(10)
    path.add_bend(np.pi / 2, 10)
    path.add_bend(-np.pi / 2, 10)
    path.add_s_bend(0.06, 60, +1)
    path.add_cubic_bezier_path(
        p0=(0, 0, 0), p1=(0, 10, 1), p2=(5, 15, 3), p3=(5, 0, 4))
    print(path._segments)


if __name__ == '__main__':
    _example()
