"""Input-related functionality.

Started 17 Feb 2025.
"""

import h5py as h5

def subhalo_data_names(par, with_parents=False, with_descendants=False):
	if par['Input']['UseSOAP']:
		return subhalo_data_names_soap(
			par, with_parents=with_parents, with_descendants=with_descendants)
	else:
		return subhalo_data_names_hbt(
			par, with_parents=with_parents, with_descendants=with_descendants)

def subhalo_data_names_soap(par, with_parents=False, with_descendants=False):
	names = [
		('Coordinates', 'InputHalos/HaloCentre'),
		('Velocities', '???'),
		('GalaxyIDs', 'InputHalos/HBTplus/TrackId'),
		('CentralFlag', 'InputHalos/IsCentral'),
		('HBTplusIDs', 'InputHalos/HaloCatalogueIndex'),
	]
	if with_parents:
		names.append(
			('ParentGalaxyIDs', 'InputHalos/HBTplus/NestedParentTrackID'))
		names.append(('Depth', 'InputHalos/HBTplus/Depth'))

	if with_descendants:
		names.append(
			('DescendantGalaxyIDs', 'InputHalos/HBTplus/DescendantTrackId'))

	if par['Sources']['Neigbours']:
		names.append(('Radii', 'BoundSubhalo/???'))

	return names

def subhalo_data_names_hbt(par, with_parents=False, with_descendants=False):
	names = [
		('Coordinates', 'ComovingMostBoundPosition'),
		('Velocities', 'PhysicalAverageVelocity'),
		('GalaxyIDs', 'TrackId'),
		('Depth', 'Depth'),
	]
	if with_parents:
		names.append(('ParentGalaxyIDs', 'NestedParentTrackId'))
	if with_descendants:
		names.append(('DescendantGalaxyIDs', 'DescendantTrackId'))
	if par['Sources']['Neighbours']:
		names.append(('Radii', 'REncloseComoving'))

	return names

def form_snapshot_file(par, isnap):
	"""Form the full snapshot file name.

	Parameters
	----------
	par : dict
		Parameter structure
	isnap : int
		Snapshot index of the snapshot file to construct.

	Returns
	-------
	file_name : str
		The full name of the snapshot file.
	"""

	# TO BE IMPLEMENTED
	snapshot_name = par['Sim']['SnapshotName']
	snapshot_name.replace('XXX', f'{isnap:04d}')
	path = par['Sim']['RootDir'] + snapshot_name
	return path


def form_subhalo_file(par, isnap):
	"""Form the full subhalo file name for a snapshot."""
	subhalo_file_name = par['Sim']['SubhaloFileName']
	subhalo_file_name.replace('XXX', f'{isnap:04d}')
	return par['Sim']['RootDir'] + subhalo_file_name

def form_subhalo_particle_file(par, isnap):
	"""Form the subhalo particle file name."""
	particle_file_name = par['Sim']['SubhaloParticleFileName']
	particle_file_name.replace('XXX', f'{isnap:04d}')
	return par['Sim']['Rootdir'] + particle_file_name

def form_subhalo_membership_file(par, isnap):
	"""Form the file name with subhalo membership information."""
	membership_file_name = par['Sim']['MembershipFileName']
	membership_file_name.replace('XXX', f'{isnap:04d}')
	return par['Sim']['Rootdir'] + membership_file_name

def load_subhalo_catalogue(sub_file, fields=[]):
	"""Load the named fields from the subhalo catalogue."""
	data = {}
	with h5.File(sub_file, 'r') as f:
		for field in fields:
			cat_name = field[0]
			internal_name = field[1]
			data[internal_name] = f[cat_name][...]

	return data

def load_subhalo_particles_external(subhalo_particle_file, base_indices):
	"""Load the structured list of particle IDs in subhaloes."""
	with h5.File(subhalo_particle_file, 'r') as f:
		ids = f['XXX']  # TO DO: look up name and implement.

	# If base_indices is specified, we need to rearrange the list so that we
	# pull the particles from the specified index into the correct location.
	if base_indices is not None: 
		ids = ids[base_indices]
	return ids

def load_subhalo_particles_nightingale(property_file, id_file):
	with h5.File(property_file, 'r') as f:
		offsets = f['Subhalo/Offset'][...]
		lengths = f['Subhalo/Lengths'][...]
	with h5.File(id_file, 'r') as f:
		ids = f['IDs'][...]

	# TO DO: CONVERT OFFSET-SEPARATED LIST TO NESTED ARRAY...

	return ids		