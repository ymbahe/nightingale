"""Input-related functionality.

Started 17 Feb 2025.
"""

import h5py as h5
import numpy as np
from pdb import set_trace

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
        ('NumberOfBoundParticles', 'Nbound'),
        ('FOF', 'HostHaloId'),
    ]
    if with_parents:
        names.append(('ParentGalaxyIDs', 'NestedParentTrackId'))
    if with_descendants:
        names.append(('DescendantGalaxyIDs', 'DescendantTrackId'))
    if par['Sources']['FreeNeighbours']:
        if par['Sources']['RadiusTypeFree'] == 'HalfMass':
            names.append(('FreeRadii', 'RHalfComoving'))
        elif par['Sources']['RadiusTypeFree'] == 'Enclosing':
            names.append(('FreeRadii', 'REncloseComoving'))

    if par['Sources']['FreeNeighbours']:
        if par['Sources']['RadiusTypeSubhaloes'] == 'HalfMass':
            names.append(('SubRadii', 'RHalfComoving'))
        elif par['Sources']['RadiusTypeSubhaloes'] == 'Enclosing':
            names.append(('SubRadii', 'REncloseComoving'))

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

    snapshot_file_name = par['Sim']['SnapshotFile']
    snapshot_file_name = snapshot_file_name.replace('XXXX', f'{isnap:04d}')
    path = par['Sim']['RootDir'] + '/' + snapshot_file_name
    # print(f"Snapshot name is '{path}'")
    return path

def form_subhalo_file(par, isnap):
    """Form the full subhalo file name for a snapshot."""
    subhalo_file_name = par['Sim']['SubhaloFile']
    subhalo_file_name = subhalo_file_name.replace('XXXX', f'{isnap:03d}')
    path = par['Sim']['RootDir'] + '/' + subhalo_file_name
    #print(f"Subhalo file name is '{path}'")
    return path

def form_subhalo_particle_file(par, isnap):
    """Form the subhalo particle file name."""
    particle_file_name = par['Sim']['SubhaloParticleFile']
    particle_file_name = particle_file_name.replace('XXXX', f'{isnap:03d}')
    path = par['Sim']['RootDir'] + '/' + particle_file_name
    #print(f"Subhalo particle file name is '{path}'")
    return path

def form_subhalo_membership_file(par, isnap):
    """Form the file name with subhalo membership information."""
    membership_file_name = par['Sim']['MembershipFile']
    membership_file_name = membership_file_name.replace('XXXX', f'{isnap:04d}')
    path = par['Sim']['Rootdir'] + '/' + membership_file_name
    #print(f"Particle membership file name is '{path}'")
    return path

def load_subhalo_catalogue_soap(sub_file, fields=[]):
    """Load the named fields from the subhalo catalogue."""
    data = {}
    with h5.File(sub_file, 'r') as f:
        for field in fields:
            cat_name = field[0]
            internal_name = field[1]
            data[internal_name] = f[cat_name][...]

    return data

def load_subhalo_catalogue_hbt(sub_file, fields=[]):
    """Load the named fields from the HBT subhalo catalogue."""
    print(f"Loading HBT subhalo data\n   [{sub_file}]")
    print("Fields to load:")
    for ifield in fields:
        print(f"{ifield[1]} --> {ifield[0]}")
    data = {}
    dtypes = {}
    shapes = {}
    with h5.File(sub_file, 'r') as f:
        n_files = f['NumberOfFiles'][0]
        n_subhaloes = f['NumberOfSubhalosInAllFiles'][0]
        for field in fields:
            key = field[0]
            dtypes[key] = f['Subhalos'][field[1]].dtype
            shapes[key] = f['Subhalos'][field[1]].shape

    for field in fields:
        key = field[0]
        if len(shapes[key]) == 1:
            data[key] = np.zeros(n_subhaloes, dtype=dtypes[key])
        else:
            full_shape = list(shapes[key])
            full_shape[0] = n_subhaloes
            data[key] = np.zeros(full_shape, dtype=dtypes[key])

    offset = 0
    for ifile in range(n_files):
        file_name = sub_file.replace('.0.hdf5', f'.{ifile}.hdf5')
        print(f"  -- reading file {ifile}...")
        with h5.File(file_name, 'r') as f:
            sub_data = f['Subhalos'][...]
            n_subhaloes_file = len(sub_data)
            for field in fields:
                key = field[1]
                name = field[0]
                print(f"    -- {key} --> {name}...")
                try:
                    data[name][offset:offset+n_subhaloes_file, ...] = (
                        sub_data[key])
                except KeyError:
                    set_trace()
        offset += n_subhaloes_file

    # Almost there. But we also need a 'CentralFlag', for compatibility
    #data['CentralFlag'] = np.zeros(offset, dtype=np.int8)
    #ind_central = np.nonzero(data['Depth'] == 0)[0]
    #data['CentralFlag'][ind_central] = 1

    # ... and the same to indicate whether the subhalo is resolved
    #data['ResolvedFlag'] = np.zeros(offset, dtype=np.int8)
    #ind_resolved = np.nonzero(data['NumberOfBoundParticles'] > 0)[0]
    #data['ResolvedFlag'][ind_resolved] = 1

    return data

def load_subhalo_particles_external(subhalo_particle_file, base_indices=None):
    """Load the structured list of particle IDs in subhaloes."""

    with h5.File(subhalo_particle_file, 'r') as f:
        n_files = f['NumberOfFiles'][0]

    ids = np.zeros(0, dtype=int)
    for ifile in range(n_files):
        file_name = subhalo_particle_file.replace('.0.hdf5', f'.{ifile}.hdf5')
        with h5.File(file_name, 'r') as f:
            curr_ids = f['SubhaloParticles'][...]
        ids = np.concatenate((ids, curr_ids))

    # If base_indices is specified, we need to rearrange the list so that we
    # pull the particles from the specified index into the correct location.
    if base_indices is not None: 
        ids = ids[base_indices]
    return ids

def load_boxsize(snapshot_file):
    """Load the boxsize from a specified snapshot file."""
    with h5.File(snapshot_file, 'r') as f:
        return f['Header'].attrs['BoxSize'][0]
