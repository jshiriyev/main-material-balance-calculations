from ._reservoir import RadialPorMed

class BaseSolver(RadialPorMed):

	def __init__(self,*args,rrock=None,fluid=None,tcomp=None):

		super().__init__(*args)

		self.rrock = rrock
		self.fluid = fluid
		self.tcomp = tcomp

	@property
	def rrock(self):
		"""Getter for the reservoir rock properties."""
		return self._rrock

	@rrock.setter
	def rrock(self,value):
		"""Setter for the reservoir rock properties."""
		self._rrock = value

	@property
	def fluid(self):
		"""Getter for the reservoir fluid properties."""
		return self._fluid

	@fluid.setter
	def fluid(self,value):
		"""Setter for the reservoir fluid properties."""
		self._fluid = value

	@property
	def tcomp(self):
		"""Getter for the total compressibility."""
		return None if self._tcomp is None else self._tcomp*6894.76

	@tcomp.setter
	def tcomp(self,value):
		"""Setter for the total compressibility."""
		if value is None:
			try:
				self._tcomp = self.rrock._comp+self.fluid._comp
			except Exception as e:
				logging.warning(f"Missing attribute when calculating total compressibility: {e}")
		else:
			self._tcomp = value/6894.76

	@property
	def hdiff(self):
		"""Getter for the hydraulic diffusivity."""
		if not hasattr(self,"_hdiff"):
			self.hdiff = None

		return self._hdiff*(3.28084**2)*(24*60*60)

	@hdiff.setter
	def hdiff(self,value):
		"""Setter for the hydraulic diffusivity."""
		self._hdiff = (self.rrock._perm)/(self.rrock._poro*self.fluid._visc*self._tcomp)

	@property
	def vpore(self):
		"""Getter for the pore volume."""
		if not hasattr(self,"_vpore"):
			self.vpore = None

		return self._vpore/(0.3048**3)

	@vpore.setter
	def vpore(self,value):
		"""Setter for the pore volume."""
		self._vpore = self._volume*self.rrock._poro
	