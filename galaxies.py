"""Galaxy population information.

Started 17 Feb 2025.
"""

class GalaxyBase:

	def dummy(self):
		print("Test")

class SnapshotGalaxies(GalaxyBase):

	def __init__(self, snapshot):
		self.snap = snapshot


class TargetGalaxy(GalaxyBase):

	def __init__(self, all_galaxies, igal):
		self.igal = igal
		self.all = all_galaxies

