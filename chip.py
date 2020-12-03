import itertools
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

from port import Port


class Cell:
    def __init__(self, name: str):
        """
        Creates a new Cell named `name` at `origin`

        :param name: Name of the cell, needs to be unique
        """
        self.name = name
        self.guides_dict = {}
        self.marker_dict = {}
        self._bounds = None

    @property
    def bounds(self):
        """
        The outer bounding box of the cell. Returns `None` if it is empty.
        """
        return self.get_bounds()

    def get_bounds(self):
        """
        Calculates and returns the envelope for the given layers.
        Returns `None` if it is empty.
        """
        def get_min_max(elem_dict: dict, index: int):
            min_value, max_value = None, None
            if bool(elem_dict):
                for key, elem in elem_dict.items():
                    tmp_max = np.max(elem[:, index])
                    tmp_min = np.min(elem[:, index])
                    if max_value is None or max_value < tmp_max:
                        max_value = tmp_max
                    if min_value is None or min_value < tmp_min:
                        min_value = tmp_min
            return min_value, max_value

        bounds = []
        if self._bounds is not None:
            return bounds
        else:
            xmin, xmax = get_min_max(self.guides_dict, 0)
            ymin, ymax = get_min_max(self.guides_dict, 1)
            mark_xmin, mark_xmax = get_min_max(self.marker_dict, 0)
            mark_ymin, mark_ymax = get_min_max(self.marker_dict, 1)

            xmax = np.max(xmax, mark_xmax)
            ymax = np.max(ymax, mark_ymax)
            xmin = np.min(xmin, mark_xmin)
            ymin = np.min(ymin, mark_ymin)
        return ((xmin, xmax), (ymin, ymax))

    @property
    def size(self):
        """
        Returns the size of the cell
        """
        bounds = self.bounds
        if bounds is None:
            return 0, 0
        else:
            return bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0]

    def merge_segments(self, guide):
        """
        Extract the segments in the waveguide and merge them in a single numpy
        array.

        :param guide: waveguide object
        :rtype: numpy array
        """

        waveguide_shape = np.array([])
        if guide.get_segments() is not None:
            for segment in guide.get_segments():
                if waveguide_shape.size == 0:
                    waveguide_shape = segment[2]
                else:
                    waveguide_shape = np.concatenate(
                        (waveguide_shape, segment[2]), axis=0)

        return waveguide_shape

    def add_guide(self, guide, num: int):
        """
        Adds a waveguide to a the chip

        :param guide: waveguide object
        :param num: index of the waveguide
        """
        if num not in self.guides_dict:
            self.guides_dict[num] = []
        self.guides_dict[num] = self.merge_segments(guide)

    def get_element_coordinates(self, elem_dict: dict):
        elem = []
        for key, array in elem_dict.items():
            elem.append(array)
        return elem

    def save_image(self, filename: str, resolution=1., ylim=(None, None), xlim=(None, None), scale=1.):
        """
           Save cell object as an image.

           You can either use a rasterized file format such as png but also formats such as SVG or PDF.

           :param filename: Name of the image file.
           :param resolution: Rasterization resolution in GDSII units.
           :param ylim: Tuple of (min_x, max_x) to export.
           :param xlim: Tuple of (min_y, max_y) to export.
           :param scale: Defines the scale of the image
           """
        import matplotlib.pyplot as plt

        # For vector graphics, map 1um to {resolution} mm instead of inch.
        is_vector = filename.split(
            '.')[-1] in ('svg', 'svgz', 'eps', 'ps', 'emf', 'pdf')
        scale *= 5 / 127. if is_vector else 1.

        fig, ax = plt.subplots()
        for guide in self.get_element_coordinates(self.guides_dict):
            ax.plot(guide[:, 0], guide[:, 1], '-b')

        for mark in self.get_element_coordinates(self.marker_dict):
            ax.plot(mark, '-k')

        # plot frame

        # Autoscale, then change the axis limits and read back what is actually displayed
        ax.autoscale(True, tight=True)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        actual_ylim, actual_xlim = ax.get_ylim(), ax.get_xlim()
        fig.set_size_inches(np.asarray(
            (actual_xlim[1] - actual_xlim[0], actual_ylim[1] - actual_ylim[0])) * scale)

        ax.set_aspect(1)
        ax.axis('off')

        fig.set_dpi(1 / resolution)
        plt.savefig(filename, transparent=True,
                    bbox_inches='tight', dpi=1 / resolution)
        plt.close()

    def show(self, padding=5):
        """
        Shows the current cell

        :param layers: List of the layers to be shown, passing None shows all layers
        :param padding: padding around the structure
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        for guide in self.get_element_coordinates(self.guides_dict):
            ax.plot(guide[:, 0], guide[:, 1], '-b')

        for mark in self.get_element_coordinates(self.marker_dict):
            ax.plot(mark, '-k')

        bounds = self.get_bounds()
        ax.set_xlim(bounds[0][0] - padding, bounds[0][1] + padding)
        ax.set_ylim(bounds[1][0] - padding, bounds[1][1] + padding)
        ax.set_aspect(1)
        plt.show()


if __name__ == '__main__':
    from port import Port
    from waveguide import Waveguide

    device_cell = Cell('my_cell')
    start_port = Port(origin=(0, 0, 0), angle=0)
    waveguide = Waveguide.make_at_port(start_port)
    for i_bend in range(2):
        waveguide.add_bend(angle=np.pi, radius=60 + i_bend * 40)
    device_cell.add_guide(waveguide, 1)
    print(device_cell.size)
    device_cell.show()
    # device_cell.save_image('chip.pdf')
    # # Creates the output file by using gdspy or fatamorgana. To use the implemented parallel processing, set
    # # parallel=True.
    # device_cell.save(name='my_design', parallel=True)

    # array_cell = Cell('Array')
    # array_cell.add_cell(device_cell, rows=2, columns=2, spacing=(1000, 1000))

    # array_cell.save()
