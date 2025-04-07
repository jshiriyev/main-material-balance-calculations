import numpy as np

class RadPorMed():
    """A reservoir class for radial flow calculations."""

    _FT_TO_METER = 0.3048

    def __init__(self,radius:float,height:float,/):
        """Initialize the base reservoir class for analytical calculations.

        Arguments:
        ---------
        radius (float): Radius of the reservoir in feet.
        height (float): Height of the reservoir in feet.

        """
        self.radius = radius
        self.height = height

    @property
    def radius(self):
        """Getter for the reservoir radius."""
        return self._radius/self._FT_TO_METER

    @radius.setter
    def radius(self,value):
        """Setter for the reservoir radius."""
        self._radius = value*self._FT_TO_METER

    @property
    def height(self):
        """Getter for the reservoir height."""
        return self._height/self._FT_TO_METER

    @height.setter
    def height(self,value):
        """Setter for the reservoir height."""
        self._height = value*self._FT_TO_METER

    @property
    def surface(self):
        """Getter for the reservoir surface area perpendicular to flow."""
        if not hasattr(self,"_surface"):
            self.surface = None

        return self._surface/(self._FT_TO_METER**2)

    @surface.setter
    def surface(self,value):
        """Setter for the reservoir surface area perpendicular to flow."""
        self._surface = np.pi*self._radius**2

    @property
    def volume(self):
        """Getter for the reservoir volume."""
        if not hasattr(self,"_volume"):
            self.volume = None

        return self._volume/(self._FT_TO_METER**3)

    @volume.setter
    def volume(self,value):
        """Setter for the reservoir volume."""
        self._volume = np.pi*self._radius**2*self._height

if __name__ == "__main__":

    res = RadPorMed(1000,20)

    print(res.radius)

    print(res.surface)

    print(res.volume)