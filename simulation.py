"""Simulation-wide properties

Started 18 Feb 2025.
"""

import numpy as np

class Simulation:
	"""Class to hold basic information about a simulation.

	This is mostly used to attach references to the currently used snapshots.
	"""

	def __init__(self, par):
		self.par = par
		self.load_redshifts(par)

	def load_redshifts(self, par):
		"""Load the redshift information for the simulation."""
		redshift_file = par['Sim']['RootDir'] + '/' + par['Sim']['RedshiftFile']
		redshifts = np.loadtxt(redshift_file)

		if par['Sim']['RedshiftsAreAexp']:
			redshifts = 1 / redshifts - 1

		self.redshifts = redshifts
		self.aexps = 1 / (redshifts + 1)