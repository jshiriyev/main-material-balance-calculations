import logging

import numpy as np

from scipy import special

from ._solver import BaseSolver
from ._result import Result

class Transient(BaseSolver):
    """Line source solution based on exponential integral."""

    def __init__(self,*args,**kwargs):

        super().__init__(*args,**kwargs)

    def __call__(self,well,pinit:float=None):

        self.well = well
        self.tmin = None
        self.tmax = None

        self.pterm = None
        self.pinit = pinit

        return self

    @property
    def well(self):
        """Getter for the well item."""
        return self._well
    
    @well.setter
    def well(self,value):
        """Setter for the well item."""
        self._well = value

    @property
    def tmin(self):
        """Getter for the minimum time limit for the transient solution because of the finite wellbore size."""
        return self._tmin/(24*60*60)
    
    @tmin.setter
    def tmin(self,value):
        """Setter for the mimimum time limit for the transient solution because of the finite wellbore size."""
        self._tmin = 100*self.well._radius**2/self._hdiff

    @property
    def tmax(self):
        """Getter for the maximum time limit for the transient solution because of the finite reservoir size."""
        return self._tmax/(24*60*60)

    @tmax.setter
    def tmax(self,tmax=None):
        """Setter for the maximum time limit for the transient solution because of the finite reservoir size."""
        self._tmax = 0.25*self._radius**2/self._hdiff

    @property
    def pterm(self):
        """Getter for the pressure term used in analytical equations."""
        return self._pterm/6894.76

    @pterm.setter
    def pterm(self,value):
        """Setter for the pressure term used in analytical equations."""
        self._pterm = (self.well._cond)/(2*np.pi*self._flow*self.fluid._mobil)

    @property
    def pinit(self):
        """Getter for the initial reservoir pressure."""
        return self._pinit/6894.76

    @pinit.setter
    def pinit(self,values):
        """Setter for the initial reservoir pressure."""
        self._pinit = np.ravel(value).astype(float)*6894.76

    def solve(self,times,nodes):
        """Solves for the pressure values at transient state."""
        times  = self.correct(times)
        result = Result(times,nodes)
        expi   = special.expi(-(result._nodes**2)/(4*self._hdiff*result._times))

        deltap = self._pterm*(-1/2*expi+self.well._skin)
        result._press = self._pinit-deltap

        return result

    def correct(self,times:np.ndarray):
        """It sets the time values to np.nan if it is outside of the solver limit."""
        bound_internal = times>=self.tmin
        bound_external = times<=self.tmax

        if np.any(~bound_internal):
            logging.warning("Not all times satisfy the early time limits!")

        if np.any(~bound_external):
            logging.warning("Not all times satisfy the late time limits!")

        valids = np.logical_and(bound_internal,bound_external)

        return np.where(valids,times,np.nan)