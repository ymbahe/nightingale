"""Simulation-wide properties

Started 18 Feb 2025.
"""

class Simulation:
	"""Class to hold basic information about a simulation.

	This is mostly used to attach references to the currently used snapshots.
	"""

	def __init__(self, par):
		self.par = par

	def load_redshifts(self, par):
		"""Load the redshift information for the simulation."""
		pass