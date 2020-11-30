import numpy as np
from helpers import normalize_phase


class Port:
    """
    Helper element representing an end of a waveguide.

    It can be used to attach some other elements and chaining parts.
    :param origin:  coordinates of the port.
    :param angle:   angle of the port.
    :type:          float
    """

    def __init__(self, origin, angle):
        self.origin = origin
        self.angle = angle

    def copy(self):
        """
        Create a copy of the port.

        :return:    Copy of the port.
        :rtype:     Port object.
        """
        return Port(self.origin, self.angle)

    def get_parameters(self):
        """
        Dictionary representation of the port property.

        :return:    A dict with ``origin``, ``angle`` of the port.
        :rtype:     dict
        """

        return {key: getattr(self, key) for key in ('origin', 'angle')}

    def set_port_properties(self, **kwargs):
        """
        Set port parameters via named keyword arguments.
        :param kwargs: The keywords to set.
        :return: The modified port
        :rtype: Port
        """
        for key, value in kwargs.items():
            assert key in (
                'origin', 'angle'), '"%s" is not a valid property' % key
            setattr(self, key, value)

        return self

    @property
    def inverted_direction(self):
        """
        Return an port which points in the opposite direction.

        :return: a copy of this port, pointing in the opposite direction.
        :rtype: port
        """
        inverted_port = self.copy()
        inverted_port.angle = inverted_port.angle + np.pi
        return inverted_port

    @property
    def origin(self):
        """
        Coordinates definition of the port.
        """
        return self._origin

    @origin.setter
    def origin(self, origin):
        assert len(origin) == 3, 'origin must be a 3D coordinate'
        self._origin = np.array(origin, dtype=float)

    @property
    def x(self):
        return self._origin[0]

    @x.setter
    def x(self, x):
        self._origin[0] = x

    @property
    def y(self):
        return self._origin[1]

    @y.setter
    def y(self, y):
        self._origin[1] = y

    @property
    def z(self):
        return self._origin[2]

    @z.setter
    def z(self, z):
        self._origin[2] = z

    # @property
    # def angle(self):
    #     """
    #     The angle of the port.
    #     """
    #     return normalize_phase(self._angle)

    # @angle.setter
    # def angle(self, angle):
    #     self._angle = angle  # % (2 * np.pi)

    def parallel_offset(self, offset):
        """
        Returns a new port, which offset in parallel from this port.

        :param offset: Offset from the center of the port. Positive is left of the port.
        :type offset: float
        :return: The new offset port
        :rtype: Port
        """
        port = self.copy()
        offset = [offset * np.cos(self.angle + np.pi / 2),
                  offset * np.sin(self.angle + np.pi / 2),
                  port.z]
        port.origin = port.origin + offset
        return port

    def longitudinal_offset(self, offset):
        """
        Returns a new port, which offset in in direction of this port.

        :param offset: Offset from the end of the port. Positive is the direction, the port is pointing.
        :type offset: float
        :return: The new offset port
        :rtype: Port
        """

        port = self.copy()
        offset = [offset * np.cos(self.angle), offset * np.sin(self.angle),
                  port.z]
        port.origin = port.origin + offset
        return port

    def rotated(self, angle):
        """
        Returns a new port, which is rotated by the given angle.

        :param angle: Angle to rotate.
        :type angle: float
        :return: The new rotated port
        :rtype: Port
        """

        port = self.copy()
        port.angle += angle
        return port
