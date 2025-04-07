class LinPorMed():
    """A reservoir class for one-dimensional linear flow calculations."""

    _FT_TO_METER = 0.3048

    def __init__(self,length:float,width:float,height:float,/):
        """Initialize the base reservoir class for analytical calculations.

        Arguments:
        ----------
        length (float): Length of the reservoir in feet.
        width (float): Width of the reservoir in feet.
        height (float): Height of the reservoir in feet.

        """
        self.length = length
        self.width  = width
        self.height = height

    @property
    def length(self):
        """Getter for the reservoir length."""
        return self._length/self._FT_TO_METER

    @length.setter
    def length(self,value):
        """Setter for the reservoir length."""
        self._length = value*self._FT_TO_METER

    @property
    def width(self):
        """Getter for the reservoir width."""
        return self._width/self._FT_TO_METER

    @width.setter
    def width(self,value):
        """Setter for the reservoir width."""
        self._width = value*self._FT_TO_METER

    @property
    def height(self):
        """Getter for the reservoir height."""
        return self._height/self._FT_TO_METER

    @height.setter
    def height(self,value):
        """Setter for the reservoir height."""
        self._height = value*self._FT_TO_METER

    @property
    def area(self):
        """Getter for the reservoir cross-sectional area parallel to flow."""
        if not hasattr(self,"_area"):
            self.area = None

        return self._area/(self._FT_TO_METER**2)

    @area.setter
    def area(self,value):
        """Setter for the reservoir cross-sectional area parallel to flow."""
        self._area = self._height*self._width

    @property
    def surface(self):
        """Getter for the reservoir surface area perpendicular to flow."""
        if not hasattr(self,"_surface"):
            self.surface = None

        return self._surface/(self._FT_TO_METER**2)

    @surface.setter
    def surface(self,value):
        """Setter for the reservoir surface area perpendicular to flow."""
        self._surface = self._length*self._width

    @property
    def volume(self):
        """Getter for the reservoir volume."""
        if not hasattr(self,"_volume"):
            self.volume = None

        return self._volume/(self._FT_TO_METER**3)

    @volume.setter
    def volume(self,value):
        """Setter for the reservoir volume."""
        self._volume = self._length*self._height*self._width

if __name__ == "__main__":

    res = Reservoir(1000,100,20)

    print(res.length)

    print(res.area)
    print(res.volume)