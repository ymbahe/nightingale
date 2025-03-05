"""Nightingale-specific IO"""

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
        self.simulation = snapshot.sim
        self.input_sub = subhaloes
        self.particles = particles
        self.n_input_subhaloes = subhaloes.n_subhaloes

        # Work out where to eventually write the output
        self.file = self.construct_output_file()

        # Work out which input subhaloes are "detected" (=in output)
        ind_found, mass_found = self.find_detected_subhaloes()
        self.n_output_subhaloes = len(ind_found)

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

        return ind_found, subhalo_masses[ind_found], subhalo_nparts[ind_found]

    def initialise_output_fields(self):
        """Initialise the output data structures."""
        n_sub = self.n_output_subhaloes
        n_fof = self.n_output_fof

        self.subhaloes = {}
        self.fof = {}

        self.comments = {
            'FOF': io_comments.fof_comments,
            'Subhaloes': io_comments.subhalo_comments,
            'IDs': io_comments.ids_comments
        }

    def assign_subhalo_ids(input_indices, masses)
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

        # Invert the translation list to get output-from-input
        inv_sorter = np.zeros(self.n_input_subhaloes, dtype=int) - 1
        inv_sorter[sorter] = np.arange(self.n_output_subhaloes)

        # Store the results
        self.output_shi_from_input_shi = inv_sorter

        self.subhaloes['InputHaloIDs'] = sorter
        self.subhaloes['FOFIndices'] = fof[sorter]
        self.subhaloes['FOFFlags'] = flag_fof[sorter]
        self.subhaloes['CentralFlags'] = flag_cen[sorter]

    def count_particles(self):
        """Compute the particle count and mass of all input subhaloes."""

        particles = self.particles
        n_input_subhaloes = self.input_sub.n_subhaloes

        # Compute total mass and particle count of each subhalo. We use
        # bincount for this. There should be no particles outside of subhaloes
        # anymore at this point, but better check...
        if np.min(particles.subhalo_indices) < 0:
            print(f"We still have particles outside subhaloes...")
            set_trace()

        num_part_by_subhalo = np.bincount(particles.subhalo_indices)
        mass_by_subhalo = np.bincount(
            particles.subhalo_indices, weights=particles.mass)

        return num_part_by_subhalo, mass_by_subhalo

    def prepare(self):
        """Get halo centres and sort particles by radii."""

        shi_in = self.subhaloes['InputHaloIDs']

        # Find the final position of each subhalo
        if self.par['COPAtUnbinding']:
            cop = input_sh.get_cop(shi_in)
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

        if par['RecordFinalUnbindingFrame']:
            self.subhaloes['FinalUnbindingCentres'] = (
                input_sh.get_coordinates(shi_in))
            self.subhaloes['FinalUnbindingVelocities'] = (
                input_sh.get_velocities(shi_in))

    def compute_cop(shi):
        """Compute the centre of potential for (input) subhaloes."""
        pass

    def compute_output_quantities():
        """Compute what we want to know about the subhaloes."""

        print("Computing galaxy properties... ", end='', flush=True)

        particles = self.particles
        ObjectFile = self.par['GalQuantObjectFile']

        # -------------------------------------------------------------
        # Get all the fields into the correct form for external C code
        # -------------------------------------------------------------

        # Metadata
        c_numPart = c.c_long(particles.n_parts)
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

        # Output fields
        cop_p = self.subhaloes['CentresOfPotential'].ctypes.data_as(c.c_void_p)
        off_p = self.subhaloes['ParticleOffsets'].ctypes.data_as(c.c_void_p)
        len_p = self.subhaloes['ParticleNumbers'].ctypes.data_as(c.c_void_p)
        lenType_p = self.subhaloes['ParticleNumbersByType'].ctypes.data_as(c.c_void_p)
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
        
        nargs = 33
        myargv = c.c_void_p * 33
        argv = myargv(c.addressof(c_numPart), 
                      c.addressof(c_numSH),
                      mass_p, pos_p, vel_p, type_p, shi_p, rad_p, ids_p,
                      cop_p, off_p, len_p, lenType_p, offType_p,
                      offTypeAp_p,
                      massTypeAp_p, vmax_p, rvmax_p, mtot_p, 
                      comPos_p, zmfVel_p, rMax_p, rMaxType_p, 
                      comPosType_p, zmfVelType_p, velDisp_p, 
                      angMom_p, axes_p, axRat_p, kappaCo_p, smr_p,
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
        output_file = self.par['OutputFile'].replace('XXX', f'{isnap:04d}')
        id_file = self.par['OutputFileIDs'].replace('XXX', f'{isnap:04d}')

        self.file = self.par['Directory'] + '/' + output_file
        self.id_file = self.par['Directory'] + '/' + id_file

    def write_header(file_name):
        """Write the header information to the specified file."""

        # Copy all parameters into a separate group in the output file
        tools.dict2att(self.sim.par, file_name, 'Parameters', bool_as_int=True)

        # Write relevant metadata to 'Header' group
        with h5.File(file_name, 'a') as o:
            o['Header'].attrs['Redshift'] = self.snapshot.redshift
            o['Header'].attrs['ExpansionFactor'] = self.snapshot.aexp
            o['Header'].attrs['NumberOfSubhaloes'] = self.n_output_subhaloes
            o['Header'].attrs['NumberOfFOFs'] = self.n_output_fof

    def write_particle_data(file_name):
        """Write the desired data for subhalo particles."""
        with h5.File(file_name, 'a') as o:
            o['IDs'] = self.particles.ids
            o['IDs'].attrs['Comment'] = self.comments['IDs']['IDs']
            

    def write_fof_info(file_name):
        """Write FOF-level information to the specified file."""
        grp = 'FOF'
        with h5.File(file_name, 'a') as o:
            for key in self.fof:
                dname = f'{grp}/{key}'
                o[dname] = self.fof[key]
                o[dname].attrs['Comment'] = self.comments['FOF'][key]

    def write_cross_indices(file_name):
        """Write cross-indices between output and input catalogues."""
        grp = 'Subhalo'

        # Subhalo --> galaxy index
        hdf5.write_data(
            file_name, grp + 'TrackID', self.subhalo['TrackID'],
            comment = "Track ID for each subhalo index, from input catalogue."
        )
        
        # Subhalo --> FOF
        hdf5.write_data(
            file_name, grp + 'FOF', self.subhalo['FOFIndices']
            comment = "Index of the FOF group to which this subhalo "
            "belongs. Note that this may not be a real FOF group, as "
            "specified by FOFFlags."
        )
        
        # Output <--> Input subhalo cross indices
        hdf5.write_data(
            file_name, grp + 'InputHalo', self.subhalo['InputHaloIDs'], 
            comment = "Index of the corresponding subhalo in "
            "the input catalogue. If this is < 0, it means "
            "that the subhalo was lost in the input catalogue but recovered "
            "by Nightingale."
        )
        hdf5.write_data(
            file_name, grp + 'IndexByInputHalo', self.output_shi_from_input_shi, 
            comment = "Reverse index to identify the subhalo "
            "corresponding to a given input subhalo. To find "
            "the (Nightingale) subhalo corresponding to "
            "input subhalo index i, look at Nightingale index "
            "IndexByInputHalo[i]."
        )

    def write_subhalo_particle_links(file_name):
        """Write the information to retrieve subhalo particles."""
        grp = 'Subhalo'

        hdf5.write_data(
            file_name, grp + 'ParticleOffsets',
            self.subhaloes['ParticleOffsets'],
            comment = "First index in the ID list belonging to "
            "subhalo i. The particles belonging to this subhalo "
            "are stored in indices [offset]:[offset]+[length]."
        )
        hdf5.write_data(
            file_name, grp + 'NumberOfParticles',
            self.subhaloes['NumberOfParticles'] 
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

    def write_subhalo_properties(file_name):
        """Write the physical subhalo properties."""
        grp = 'Subhalo'  


def initialize_hdf5_file(file_name):
    if os.path.isfile(file_name):
        os.rename(file_name, file_name + '.bak')
