"""Nightingale-specific IO"""

import h5py as h5
import numpy as np
from pdb import set_trace
import iocomments
import os
import tools
import hdf5
import ctypes as c

def form_nightingale_property_file(par, isnap):
    """Form the property file name for a given snapshot."""
    catalogue_name = par['Output']['CatalogueName']
    catalogue_name = catalogue_name.replace('XXXX', f'{isnap:04d}')
    return par['Output']['Directory'] + catalogue_name

def form_nightingale_id_file(par, isnap):
    """Form the subhalo ID file name for a given snapshot."""
    id_name = par['Output']['IDFileName']
    id_name = id_name.replace('XXXX', f'{isnap:04d}')
    return par['Output']['Directory'] + id_name

def form_nightingale_waitlist_file(par, isnap):
    """Form the waitlist ID file name for a given snapshot."""
    file_name = par['Output']['WaitlistIDFileName']
    file_name = file_name.replace('XXXX', f'{isnap:04d}')
    return par['Output']['Directory'] + file_name

def load_subhalo_particles_nightingale(property_file, id_file):
    with h5.File(property_file, 'r') as f:
        offsets = f['Subhalo/ParticleOffsets'][...]
        lengths = f['Subhalo/NumberOfParticles'][...]
    with h5.File(id_file, 'r') as f:
        ids = f['IDs'][...]

    n_subhaloes = len(lengths)
    ids_all = []
    for ish in range(n_subhaloes):
        ids_sh = ids[offsets[ish] : offsets[ish] + lengths[ish]]
        ids_all.append(ids_sh)

    return ids_all

def load_subhalo_catalogue_nightingale(property_file, with_descendants=False):
    names = [
        ('GalaxyIDs', 'Subhalo/TrackID'),
        ('InputHaloIndices', 'Subhalo/InputHalo'),
        ('Depth', 'Subhalo/Depths'),
        ('ParentList', 'Subhalo/Parents'),
    ]

    if with_descendants:
        names.append(('DescendantGalaxyIDs', 'Subhalo/DescendantTrackIDs'))

    data = {}
    with h5.File(property_file, 'r') as f:
        for field in names:
            cat_name = field[1]
            internal_name = field[0]
            data[internal_name] = f[cat_name][...]

    return data

def load_waitlist_particles_nightingale(waitlist_file):
    """Load the particle IDs from the waiting list."""
    with h5.File(waitlist_file, 'r') as f:
        offsets = f['Offsets'][...]
        ids = f['IDs'][...]

    waitlist = []
    n_subhaloes = len(offsets) - 1
    for ish in range(n_subhaloes):
        ids_sh = ids[offsets[ish] : offsets[ish+1]]
        waitlist.append(ids_sh)

    return waitlist

def write_waitlist_particles(waitlist_file, ids, offsets):
    """Write the waitlist information."""
    with h5.File(waitlist_file, 'w') as f:
        f['Offsets'] = offsets
        f['IDs'] = ids

class Output:
    """Class for storing and writing one snapshot's output from Nightingale."""
    def __init__(self, par, snapshot, subhaloes, particles):
        """Class constructor.

        Parameters
        ----------
        par : dict
            The parameter structure
        snapshot : Snapshot instance
            The snapshot for which we are initialising output.
        """
        self.par = par['Output']
        self.verbose = par['Verbose']
        self.snapshot = snapshot
        self.sim = snapshot.sim
        self.input_sub = subhaloes
        self.particles = particles
        self.n_input_subhaloes = subhaloes.n_input_subhaloes

        # Work out where to eventually write the output
        self.construct_output_files()

        # Work out which input subhaloes are "detected" (=in output)
        ind_found, mass_found = self.find_detected_subhaloes()
        self.n_output_subhaloes = len(ind_found)
        self.n_output_fof = np.count_nonzero(
            self.snapshot.subhaloes.centrals[ind_found] == ind_found)
        print(f"There are {self.n_output_subhaloes} subhaloes and "
              f"{self.n_output_fof} FOF groups (including fake ones).")
        
        # Set up the internal arrays that will hold the output
        self.initialise_output_fields()

        # Build the list of output subhaloes. This sets up the translation
        # arrays from input->output and output->input subhalo IDs, and 
        # records basic subhalo properties (FOF membership, cen/sat status).
        self.assign_subhalo_ids(ind_found, mass_found)


    def find_detected_subhaloes(self):
        """Find detected subhaloes and their masses"""

        # Count subhalo particles and masses
        subhalo_nparts, subhalo_masses = self.count_particles()

        # Determine which subhaloes have enough particles to be 'detected'
        min_npart = self.par['MinimumNumberOfParticles']
        ind_found = np.nonzero(subhalo_nparts >= min_npart)[0]
        n_found = len(ind_found)
        print(f"Out of {self.n_input_subhaloes}, {n_found} are in the output.")

        return ind_found, subhalo_masses[ind_found]

    def initialise_output_fields(self):
        """Initialise the output data structures."""
        n_sub = self.n_output_subhaloes
        #n_fof = self.n_output_fof

        self.subhaloes = {
            'CentresOfPotential': None,
            'ParticleOffsets': np.zeros(n_sub + 1, dtype=np.int64),
            'NumberOfParticles': np.zeros(n_sub, dtype=np.int32),
            'NumberOfParticlesByType': np.zeros((n_sub, 6), dtype=np.int32),
            'ParticleOffsetsByType': np.zeros((n_sub, 7), dtype=np.int64),
            'ParticleOffsetsByAperture': np.zeros(
                (n_sub, 6, 5), dtype=np.int64) - 1,
            'TotalMasses': np.zeros(n_sub, dtype=np.float32),
            'MassesByType': np.zeros((n_sub, 6, 5), dtype=np.float32),
            'MaximumCircularVelocities': np.zeros(
                n_sub, dtype=np.float32) + np.nan,
            'RadiiOfVMax': np.zeros(
                n_sub, dtype=np.float32) + np.nan,
            'CentresOfMass': np.zeros((n_sub, 3), dtype=np.float32) + np.nan,
            'MassAveragedVelocities': np.zeros((n_sub, 3), dtype=np.float32) + np.nan,
            'CentresOfMassByType': np.zeros((n_sub, 6, 3), dtype=np.float32) + np.nan,
            'MassAveragedVelocitiesByType': np.zeros((n_sub, 9, 3), dtype=np.float32) + np.nan,
            'MaximumRadii': np.zeros(n_sub, dtype=np.float32) + np.nan,
            'MaximumRadiiByType': np.zeros((n_sub, 6), dtype=np.float32) + np.nan,
            'VelocityDispersions': np.zeros((n_sub, 6), dtype=np.float32) + np.nan,
            'AngularMomenta': np.zeros((n_sub, 9, 3), dtype=np.float32) + np.nan,
            'PrincipalAxes': np.zeros((n_sub, 6, 3, 3), dtype=np.float32) + np.nan,
            'PrincipalAxisRatios': np.zeros((n_sub, 6, 2), dtype=np.float32) + np.nan,
            'KappaCo': np.zeros((n_sub, 2), dtype=np.float32) + np.nan,
            'StellarRadii': np.zeros((n_sub, 2, 3), dtype=np.float32) + np.nan,
            'TotalHalfMassRadii': np.zeros(n_sub, dtype=np.float32) + np.nan
        }
        self.fof = {}

        self.comments = {
            'FOF': iocomments.fof_comments,
            'Subhaloes': iocomments.subhalo_comments,
            'IDs': iocomments.ids_comments
        }

    def assign_subhalo_ids(self, input_indices, masses):
        """Establish the output order of subhaloes.
        
        All galaxies to be included in the output are sorted, first by
        FOF-index and second by mass (in descending order). The central galaxy
        is always output first (provided it has not been lost).
        
        Writes:
        --------
            self.output_shi_from_input_shi
            self.subhaloes (add keys)
        """

        # Connect each (detected) subhalo to a (real or fake) FOF
        fof, flag_fof = self.input_sub.fof_from_subhalo_index(input_indices)

        # Work out which subhaloes are centrals. If we want to force them to
        # appear first in the subhalo list for their subhalo, assign them
        # infinite mass (only used for ordering!)
        flag_cen = np.zeros(self.n_output_subhaloes, dtype=np.int8)
        ind_cen = np.nonzero(
            self.input_sub.centrals[input_indices] == input_indices)[0]
        flag_cen[ind_cen] = 1
        if self.par['ListCentralsFirst']:
            masses[ind_cen] = np.inf

        # Create a double sorter by FOF and mass, which is effectively the
        # translation from OutputID to InputID.
        sorter = np.lexsort((-masses, fof))
        if len(sorter) != self.n_output_subhaloes:
            print(f"Inconsistent array lengths!")
            set_trace()
        input_from_output = input_indices[sorter]
            
        # Invert the translation list to get output-from-input
        output_from_input = np.zeros(self.n_input_subhaloes, dtype=np.int32) - 1
        output_from_input[input_from_output] = np.arange(len(input_indices))

        # Store the results
        self.output_shi_from_input_shi = output_from_input

        self.subhaloes['InputHaloIDs'] = input_from_output
        self.subhaloes['FOFIndices'] = fof[sorter]
        self.subhaloes['FOFFlags'] = flag_fof[sorter]
        self.subhaloes['CentralFlags'] = flag_cen[sorter]
        self.subhaloes['GalaxyIDs'] = (
            self.input_sub.galaxy_ids[input_from_output])
        self.subhaloes['Depths'] = self.input_sub.depth[input_from_output]
        self.subhaloes['DescendantGalaxyIDs'] = (
            self.input_sub.descendant_galaxy_ids[input_from_output])
        
        # For the parent list we have to translate the entries from internal
        # to output indices, ignoring any non-existing ones (negative)
        out_parents = self.input_sub.parent_list[input_from_output, :]
        ind_real = np.nonzero(out_parents >= 0)
        out_parents[ind_real[0], ind_real[1]] = (
            output_from_input[out_parents[ind_real[0], ind_real[1]]])
        self.subhaloes['Parents'] = out_parents
        
    def count_particles(self):
        """Compute the particle count and mass of all input subhaloes."""

        particles = self.particles
        n_input_subhaloes = self.input_sub.n_input_subhaloes

        # Compute total mass and particle count of each subhalo. We use
        # bincount for this. There should be no particles outside of subhaloes
        # anymore at this point, but better check...
        if np.min(particles.subhalo_indices) < 0:
            print(f"We still have particles outside subhaloes...")
            set_trace()

        num_part_by_subhalo = np.bincount(particles.subhalo_indices)
        mass_by_subhalo = np.bincount(
            particles.subhalo_indices, weights=particles.masses)

        return num_part_by_subhalo, mass_by_subhalo

    def prepare(self):
        """Get halo centres and sort particles by radii."""

        shi_in = self.subhaloes['InputHaloIDs']

        # Find the final position of each subhalo
        if self.par['COPAtUnbinding']:
            cop = self.input_sub.get_coordinates(shi_in)
        else:
            cop = self.compute_cop(shi_in)

        # Compute the radii of each (member) particle from their SH centre.
        # Recall that particles memberships are already updated to output SHI.
        self.particles.calculate_radii(cop)

        # Arrange the particles by subhalo and radius
        # This is *almost* their final order; they are re-split by 
        # particle type within the secondary property calculation.
        self.particles.rearrange_for_output()

        # Store the result
        self.subhaloes['CentresOfPotential'] = cop

        if self.par['RecordFinalUnbindingFrame']:
            self.subhaloes['FinalUnbindingCentres'] = (
                self.input_sub.get_coordinates(shi_in))
            self.subhaloes['FinalUnbindingVelocities'] = (
                self.input_sub.get_velocities(shi_in))

    def compute_cop(self, shi):
        """Compute the centre of potential for (input) subhaloes."""
        print(f"At-output COP calculation is not yet implemented!!")
        set_trace()

    def compute_output_quantities(self):
        """Compute what we want to know about the subhaloes."""

        print("Computing galaxy properties... ", end='', flush=True)

        particles = self.particles
        ObjectFile = self.par['GalQuantObjectFile']

        # -------------------------------------------------------------
        # Get all the fields into the correct form for external C code
        # -------------------------------------------------------------

        # Metadata
        c_numPart = c.c_long(particles.n_parts)
        print(f"particles.n_parts = {particles.n_parts}")
        print(f"Num in 2053: {np.count_nonzero(particles.subhalo_indices == 2053)}")
        c_numSH = c.c_int(self.n_output_subhaloes)
        c_verbose = c.c_int(self.verbose)
        c_epsilon = c.c_float(self.snapshot.epsilon)

        # Particle input data (must provide all as they will be re-ordered)
        mass_p = particles.masses.ctypes.data_as(c.c_void_p)
        pos_p = particles.coordinates.ctypes.data_as(c.c_void_p)
        vel_p = particles.velocities.ctypes.data_as(c.c_void_p)
        type_p = particles.ptypes.ctypes.data_as(c.c_void_p)
        shi_p = particles.subhalo_indices.ctypes.data_as(c.c_void_p)
        rad_p = particles.radii.ctypes.data_as(c.c_void_p)
        ids_p = particles.ids.ctypes.data_as(c.c_void_p)

        if np.min(particles.subhalo_indices) < 0: set_trace()
        
        # Output fields
        cop_p = self.subhaloes['CentresOfPotential'].ctypes.data_as(c.c_void_p)
        off_p = self.subhaloes['ParticleOffsets'].ctypes.data_as(c.c_void_p)
        len_p = self.subhaloes['NumberOfParticles'].ctypes.data_as(c.c_void_p)
        lenType_p = self.subhaloes['NumberOfParticlesByType'].ctypes.data_as(c.c_void_p)
        offType_p = self.subhaloes['ParticleOffsetsByType'].ctypes.data_as(c.c_void_p)
        offTypeAp_p = self.subhaloes['ParticleOffsetsByAperture'].ctypes.data_as(c.c_void_p)
        massTypeAp_p = self.subhaloes['MassesByType'].ctypes.data_as(c.c_void_p)
        vmax_p = self.subhaloes['MaximumCircularVelocities'].ctypes.data_as(c.c_void_p)
        rvmax_p = self.subhaloes['RadiiOfVMax'].ctypes.data_as(c.c_void_p)
        mtot_p = self.subhaloes['TotalMasses'].ctypes.data_as(c.c_void_p)
        comPos_p = self.subhaloes['CentresOfMass'].ctypes.data_as(c.c_void_p)
        zmfVel_p = self.subhaloes['MassAveragedVelocities'].ctypes.data_as(c.c_void_p)
        rMax_p = self.subhaloes['MaximumRadii'].ctypes.data_as(c.c_void_p)
        rMaxType_p = self.subhaloes['MaximumRadiiByType'].ctypes.data_as(c.c_void_p)
        comPosType_p = self.subhaloes['CentresOfMassByType'].ctypes.data_as(c.c_void_p)
        zmfVelType_p = self.subhaloes['MassAveragedVelocitiesByType'].ctypes.data_as(c.c_void_p)
        velDisp_p = self.subhaloes['VelocityDispersions'].ctypes.data_as(c.c_void_p)
        angMom_p = self.subhaloes['AngularMomenta'].ctypes.data_as(c.c_void_p)
        axes_p = self.subhaloes['PrincipalAxes'].ctypes.data_as(c.c_void_p)
        axRat_p = self.subhaloes['PrincipalAxisRatios'].ctypes.data_as(c.c_void_p)
        kappaCo_p = self.subhaloes['KappaCo'].ctypes.data_as(c.c_void_p)
        smr_p = self.subhaloes['StellarRadii'].ctypes.data_as(c.c_void_p)
        rhalf_p = self.subhaloes['TotalHalfMassRadii'].ctypes.data_as(c.c_void_p)
        
        nargs = 34
        myargv = c.c_void_p * 34
        argv = myargv(c.addressof(c_numPart), 
                      c.addressof(c_numSH),
                      mass_p, pos_p, vel_p, type_p, shi_p, rad_p, ids_p,
                      cop_p, off_p, len_p, lenType_p, offType_p,
                      offTypeAp_p,
                      massTypeAp_p, vmax_p, rvmax_p, mtot_p, 
                      comPos_p, zmfVel_p, rMax_p, rMaxType_p, 
                      comPosType_p, zmfVelType_p, velDisp_p, 
                      angMom_p, axes_p, axRat_p, kappaCo_p, smr_p, rhalf_p,
                      c.addressof(c_verbose), c.addressof(c_epsilon))

        lib = c.cdll.LoadLibrary(ObjectFile)
        succ = lib.galquant(nargs, argv)

    def write(self):
        """Write the output to disk."""
        
        # Initialize the output file
        initialize_hdf5_file(self.file)
        initialize_hdf5_file(self.id_file)

        # Write header information to both catalogue and ID files
        self.write_header(self.file)
        self.write_header(self.id_file)

        # Write particle-level information
        self.write_particle_data(self.id_file)

        # Write (minimal) FOF-level information
        self.write_fof_info(self.file)

        # Write cross-indices between output and input catalogues
        self.write_cross_indices(self.file)

        # Write links from subhalo to particle catalogues
        self.write_subhalo_particle_links(self.file)

        # Write physical subhalo properties
        self.write_subhalo_properties(self.file)

    def construct_output_files(self):
        isnap = self.snapshot.isnap
        self.file = self.snapshot.nightingale_property_file
        self.id_file = self.snapshot.nightingale_id_file

    def write_header(self, file_name):
        """Write the header information to the specified file."""

        # Copy all parameters into a separate group in the output file
        tools.dict2att(self.sim.par, file_name, 'Parameters', bool_as_int=True)

        # Write relevant metadata to 'Header' group
        with h5.File(file_name, 'a') as o:
            try:
                hdr = o['Header']
            except KeyError:
                o.create_group('Header')
            o['Header'].attrs['Redshift'] = self.snapshot.redshift
            o['Header'].attrs['ExpansionFactor'] = self.snapshot.aexp
            o['Header'].attrs['NumberOfSubhaloes'] = self.n_output_subhaloes
            o['Header'].attrs['NumberOfFOFs'] = self.n_output_fof

    def write_particle_data(self, file_name):
        """Write the desired data for subhalo particles."""
        with h5.File(file_name, 'a') as o:
            o['IDs'] = self.particles.ids
            o['IDs'].attrs['Comment'] = self.comments['IDs']['IDs']
            

    def write_fof_info(self, file_name):
        """Write FOF-level information to the specified file."""
        grp = 'FOF'
        with h5.File(file_name, 'a') as o:
            for key in self.fof:
                dname = f'{grp}/{key}'
                o[dname] = self.fof[key]
                o[dname].attrs['Comment'] = self.comments['FOF'][key]

    def write_cross_indices(self, file_name):
        """Write cross-indices between output and input catalogues."""
        grp = 'Subhalo/'

        # Subhalo --> galaxy index
        hdf5.write_data(
            file_name, grp + 'TrackID', self.subhaloes['GalaxyIDs'],
            comment = "Track ID for each subhalo index, from input catalogue."
        )

        # Subhalo --> parents
        hdf5.write_data(
            file_name, grp + 'Parents', self.subhaloes['Parents'],
            comment = "Parent subhaloes for each subhalo in the catalogue. "
            "The second index corresponds to the depth in the subhalo "
            "hierarchy, so 0 gives the top-level parent of each subhalo."
        )
        hdf5.write_data(
            file_name, grp + 'Depths', self.subhaloes['Depths'],
            comment = "Depth of each subhalo in the hierarchy. 0 means that "
            "it is a central, 1 a top-level satellite, and >= 2 refers to "
            "satellites of satellites. Note that this is based on the "
            "hierarchy at infall and may not be physically meaningful, "
            "for example a subhalo with a formal depth of 2 may well have "
            "become unbound from its immediate parent since infall and now "
            "really be an 'independent' (depth 1) subhalo, and vice versa."
        )

        # Subhalo --> DescendantGalaxy
        hdf5.write_data(
            file_name, grp + 'DescendantTrackIDs',
            self.subhaloes['DescendantGalaxyIDs'],
            comment="Track IDs of the descendant of each subhalo in the next "
            "snapshot."
        )
        
        # Subhalo --> FOF
        hdf5.write_data(
            file_name, grp + 'FOF', self.subhaloes['FOFIndices'],
            comment = "Index of the FOF group to which this subhalo "
            "belongs. Note that this may not be a real FOF group, as "
            "specified by FOFFlags."
        )
        
        # Output <--> Input subhalo cross indices
        hdf5.write_data(
            file_name, grp + 'InputHalo', self.subhaloes['InputHaloIDs'], 
            comment = "Index of the corresponding subhalo in "
            "the input catalogue. If this is < 0, it means "
            "that the subhalo was lost in the input catalogue but recovered "
            "by Nightingale."
        )
        hdf5.write_data(
            file_name, grp + 'IndexByInputHalo',
            self.output_shi_from_input_shi, 
            comment = "Reverse index to identify the subhalo "
            "corresponding to a given input subhalo. To find "
            "the (Nightingale) subhalo corresponding to "
            "input subhalo index i, look at Nightingale index "
            "IndexByInputHalo[i]."
        )

    def write_subhalo_particle_links(self, file_name):
        """Write the information to retrieve subhalo particles."""
        grp = 'Subhalo/'

        hdf5.write_data(
            file_name, grp + 'ParticleOffsets',
            self.subhaloes['ParticleOffsets'],
            comment = "First index in the ID list belonging to "
            "subhalo i. The particles belonging to this subhalo "
            "are stored in indices [offset]:[offset]+[length]."
        )
        hdf5.write_data(
            file_name, grp + 'NumberOfParticles',
            self.subhaloes['NumberOfParticles'] ,
            comment = "Number of particles in ID list belonging "
            "to subhalo i. The particles belonging to this subhalo "
            "are stored in indices [offset]:[offset]+[length]."
        )
        hdf5.write_data(
            file_name, grp + 'ParticleOffsetsByType',
            self.subhaloes['ParticleOffsetsByType'],
            comment = "First index in ID list belonging to "
            "subhalo i (first index) and type j (second index). "
            "The particles of this type belonging to this subhalo "
            "are stored in indices [offset_i]:[offset_i+1]. "
            "The last element (j = 6) serves as a coda for this purpose."
        )
        hdf5.write_data(
            file_name, grp + 'NumberOfParticlesByType',
            self.subhaloes['NumberOfParticlesByType'],
            comment = "Number of particles in ID list that belong "
            "to subhalo i (first index) and have type j (second "
            "index). These particles are stored at indices "
            "[offset_i]:[offset_i+1]."
        )

    def write_subhalo_properties(self, file_name):
        """Write the physical subhalo properties."""
        grp = 'Subhalo'  


def initialize_hdf5_file(file_name):
    if os.path.isfile(file_name):
        os.rename(file_name, file_name + '.bak')
