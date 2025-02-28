"""Nightingale-specific IO"""

def form_nightingale_property_file(par, isnap):
    """Form the property file name for a given snapshot."""
    catalogue_name = par['Output']['CatalogueName']
    catalogue_name.replace('XXX', f'{isnap:04d}')
    return par['Output']['Directory'] + catalogue_name


def form_nightingale_id_file(par, isnap):
    """Form the subhalo ID file name for a given snapshot."""
    id_name = par['Output']['IDFileName']
    id_name.replace('XXX', f'{isnap:04d}')
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
        self.input_sub = subhaloes
        self.particles = particles

    def process_subhaloes_and_membership():
        """Process the particle-subhalo links to outputtable form."""
        pass

    def compute_secondary_quantities():
        """Compute halo properties beyond pure particle membership."""
        pass

    def write():
        """Write the output to disk."""
        pass       




class SnapshotOutput:
    """Class for storing and writing one snapshot's output from Cantor."""

    def __init__(self, snap, order_cens_first, full_snapshot=True,
                 outloc=None):
        """
        Constructor for the class.

        Parameters
        ----------
        snap : Snapshot class instance
            The snapshot to which the output belongs.
        order_cens_first : bool
            If True, the subhaloes will be sorted so that the cen is 
            always first. Requires that central information is available.
        full_snapshot : bool, optional
            If True, some extra output will be computed and written that 
            is only relevant for snapshots (such as SF-cross-indices).
            Default is True.
        outloc : str, optional
            The output file to write the results to. If None (default),
            this is the same as the simulation-wide output file.
        """

        self.snap = snap
        self.sim = snap.sim
        self.order_cens_first = order_cens_first
        self.full_snapshot = full_snapshot

        if full_snapshot:
            self.sim_output = self.sim.output

        # Initialize internal time-stamp 
        self.timeStamp = TimeStamp()

        # Retrieve the Cantor output file for this simulation:
        if outloc is None:
            self.outloc = ht.clone_dir(self.sim.hldir) + par['Output']['File']
        else:
            self.outloc = outloc

        if self.full_snapshot:
            # Initialize last-written snapshot 
            if par['Input']['FromCantor'] and par['Snaps']['Start'] > 0:
                # Last-written is the one before the start!
                self.lastSnap = par['Snaps']['Start'] - 1
            else:
                #  -1 --> not even 0 written
                self.lastSnap = -1

        snap.output = self
        self.timeStamp.set_time('Setup')

    def __del__(self):
        """Destructor, to minimize memory leaks"""
        gc.collect()
                                                    # Class: SnapshotOutput
    def setup_snapshot_outputs(self):
        """Set up snapshot-specific output arrays."""

        # Find number of subhaloes in this snapshot
        nSH = self.nSH

        # Particle offset and length information
        self.sh_offset = np.zeros(nSH, dtype = np.int64) - 1
        self.sh_length = np.zeros(nSH, dtype = np.int32) - 1
        self.sh_lengthType = np.zeros((nSH, 6), dtype = np.int32)
        self.sh_offsetType = np.zeros((nSH, 7), dtype = np.int64)
        self.sh_offsetTypeAp = np.zeros((nSH, 6, 4), dtype = np.int64) - 1
        
        # Mass by particle type and apertures
        self.sh_massTypeAp = np.zeros((nSH, 6, 5), dtype = np.float32)
        self.sh_mass = np.zeros(nSH, dtype = np.float32)

        # Centre-of-mass position and velocity:
        self.sh_comPos = np.zeros((nSH, 3), dtype = np.float32) + np.nan
        self.sh_zmfVel = np.zeros((nSH, 3), dtype = np.float32) + np.nan
        self.sh_comPosType = np.zeros((nSH, 6, 3), dtype = np.float32) + np.nan
        self.sh_zmfVelType = np.zeros((nSH, 9, 3), dtype = np.float32) + np.nan

        # Velocity dispersions: 
        self.sh_velDisp = np.zeros((nSH, 6), dtype = np.float32) + np.nan
        
        # Maximum circular velocity and its radius:
        self.sh_vmax = np.zeros(nSH, dtype = np.float32) + np.nan
        self.sh_radOfVmax = np.zeros(nSH, dtype = np.float32) + np.nan

        # Maximum radii:
        self.sh_rMax = np.zeros(nSH, dtype = np.float32) + np.nan
        self.sh_rMaxType = np.zeros((nSH, 6), dtype = np.float32) + np.nan

        # Angular momentum vectors:
        self.sh_angMom = np.zeros((nSH, 9, 3), dtype = np.float32) + np.nan

        # MIT axes and ratios:
        self.sh_axes = np.zeros((nSH, 6, 3, 3), dtype = np.float32) + np.nan
        self.sh_axRat = np.zeros((nSH, 6, 2), dtype = np.float32) + np.nan

        # Stellar kinematic morphology parameters (Correa+17):
        self.sh_kappaCo = np.zeros((nSH, 2), dtype = np.float32) + np.nan
        
        # Stellar mass percentile radii:
        self.sh_smr = np.zeros((nSH, 2, 3), dtype = np.float32) + np.nan
        
        self.timeStamp.set_time('Set up outputs')


    def compute_cantorID(self):                   # Class: SnapshotOutput
        """
        Establish the subhalo order of all galaxies in current snap.

        For each galaxy, the total mass is computed, and they are then 
        sorted first by FOF-index and second by mass (in descending order).
        However, the central galaxy is always output first (provided its
        mass is > 0, of course).
        
        Writes:
        --------
            self.(nSH, sh_galID, sh_fof, fof_fsh, fof_nsh, fof_cenSH)
        """

        print("Computing Cantor subhalo IDs ", end='', flush = True)

        snap = self.snap
        particles = snap.particles
        galaxies = snap.target_galaxies

        if self.full_snapshot:
            cenGal = galaxies.cenGalAssoc        

        # Compute total mass of each galaxy and find those with m > 0
        df = pd.DataFrame({'mass':particles.mass, 'galaxy':particles.galaxy})
        gal_mass = df.groupby('galaxy')['mass'].sum()

        gal_found_mass = gal_mass.values
        ind_found = gal_mass.index.values

        # We need to make sure we exclude fake '-1' galaxies!
        ind_real = np.nonzero(ind_found >= 0)[0]
        gal_found_mass = gal_found_mass[ind_real]
        ind_found = ind_found[ind_real]

        self.nSH = len(ind_found)

        print("(N_SH={:d})... " .format(self.nSH), end = '', flush = True)

        if self.full_snapshot:
            # Establish FOF-index of each identified galaxy
            fof = snap.fof_index[snap.shi[cenGal[ind_found]]]

            if self.order_cens_first:
                # Set mass of centrals to infinity, to make sure they are 
                # listed first in each FOF group
                ind_cen = np.nonzero(
                    snap.sim.satFlag[ind_found, snap.isnap] == 0)[0]
                gal_found_mass[ind_cen] = np.inf

            # Create double-sorter. This is essentially the translation array
            # CantorID --> GalID, which we'll invert then.
            sorter = np.lexsort((-gal_found_mass, fof))
            
        else:
            sorter = np.argsort(-gal_found_mass)

        self.sh_galID = ind_found[sorter]
        self.gal_cantorID = yb.create_reverse_list(self.sh_galID, 
                                                   maxval = snap.sim.numGal-1)

        # Set up look-up table for particles by cantorID
        self.part_lut_galaxy = SplitList(particles.galaxy, self.sim.gal_lims) 

        if self.full_snapshot:
            # Get FOF-subhalo offsets and lengths
            self.sh_fof = snap.fof_index[snap.shi[cenGal[self.sh_galID]]]
            sh_lut_fof = SplitList(self.sh_fof, 
                                   np.arange(snap.nFOF+1, dtype = np.int))

            self.fof_fsh = sh_lut_fof.splits
            self.fof_nsh = self.fof_fsh[1:] - self.fof_fsh[:-1]

            # For convenience, also create a duplicate of FSH that is set to 
            # -1 for FOFs without any galaxies:
            self.fof_cenSH = np.copy(self.fof_fsh[:-1])
            ind_fof_noSH = np.nonzero(self.fof_nsh == 0)[0]
            self.fof_cenSH[ind_fof_noSH] = -1

        self.timeStamp.set_time('Sort subhaloes')
        print("done (took {:.3f} sec.)."
              .format(self.timeStamp.get_time()))
        
    def validate_subhaloes(self):                 # Class: SnapshotOutput
        """Validate that number of subhaloes matches expectations."""
      
        snap = self.snap
        particles = snap.particles
        galaxies = snap.target_galaxies

        # Extract resuscitated/lost galaxies from their sets
        resuscitated = np.array(list(galaxies.resuscitated))
        lost_cens = galaxies.lost_cens
        lost_sats = galaxies.lost_sats
        eliminated = particles.lost_gals

        # Find total number of galaxies tried for resuscitation
        n_resusc_cand = np.count_nonzero(
            snap.shi[galaxies.candidate_galaxies] < 0)

        print("Snap {:d}: lost {:d} cens and {:d} sats, {:d} eliminated."
              .format(snap.isnap, len(lost_cens), len(lost_sats), 
                      len(eliminated)))
        if n_resusc_cand:
            print("   Resuscitated {:d} galaxies (out of {:d} candidates, "
                  "={:.2f}%)"
            .format(len(resuscitated), n_resusc_cand, 
                    len(resuscitated)/n_resusc_cand*100))
        
        # Now verify that numbers match up...
        diff = (self.nSH + len(lost_cens) + len(lost_sats) + len(eliminated)
                + len(galaxies.spectres)
                - len(resuscitated) - snap.nSH)

        if diff:
            print("Inconsistent numbers of subhaloes (diff={:d})."
                  "Please investigate." .format(diff))
            set_trace()

    def compute_centre_and_radii(self):           # Class: SnapshotOutput
        """
        Compute/extract centre-of-potential of each galaxy, and the radii
        of all particles from these centres.
        """
        
        print("Get subhalo centres and particle radii... ", end = '',
              flush = True)

        particles = self.snap.particles 
        galaxies = self.snap.target_galaxies

        # Find centre-of-potential, depending on settings, and particle 
        # distances from this centre:
        if par['Output']['COPAtUnbinding']:
            galCen = galaxies.get_potMin_pos_multi(self.sh_galID)
        else:
            # Re-compute COP now, based on particles NOW in galaxy:
            galCen = particles.get_potMin_pos_multi(self.sh_galID)
        self.sh_centreOfPotential = galCen

        # Calculate particle radii
        particles.calculate_all_radii()

        # Also record final frame coordinates from MONK:
        self.sh_monkPos = galaxies.get_pos_multi(self.sh_galID)
        self.sh_monkVel = galaxies.get_vel_multi(self.sh_galID)

        self.timeStamp.set_time('Compute centre and radii...')
        print("done (took {:.3f} sec.)."
              .format(self.timeStamp.get_time()))

    def arrange_particles(self):                  # Class: SnapshotOutput
        """
        Establish the near- ordering of particles.

        Particles are sorted by FOF --> subhalo --> radius.
        (Particles not in a subhalo are put at the end of each FOF block,
        and are not sorted by radius). The particle list is finally
        re-ordered at a later step, where it is split by type.
        """

        print("Re-arrange particle sequence... ", end = '', flush = True)

        snap = self.snap
        particles = snap.particles

        if self.full_snapshot:
            # Establish FOF-index of each particle
            part_fof = snap.fof_index[snap.shi[particles.fof_cenGal]]
        
            # Set up by-FOF lookup table 
            # (go via fof_cenGal in case of SF-dead galaxies)
            part_lut_fof = SplitList(part_fof, 
                                     np.arange(snap.nFOF+1, dtype = np.int))
        
            # First, we need the 'boundaries' of the lookup table directly:
            self.fof_offset = part_lut_fof.splits
            self.fof_length = self.fof_offset[1:] - self.fof_offset[:-1]
        
            if (self.fof_offset[0] != 0 or 
                self.fof_offset[-1] != len(particles.fof_cenGal)):
                print("Unexpected FOF split beginning/end values!")
                set_trace()

        # Set up a list to sort IDs into output order: sort by
        # [FOF] --> subhalo --> type --> radius
        particleSortList = np.zeros_like(particles.ids)-1

        # Set up array of SHI per particle. Those not in SH are assigned
        # a SHI one above the maximum, so they will be sorted to the end 
        # of their FOF group.
        part_shi = self.gal_cantorID[particles.galaxy]
        ind_not_in_sh = np.nonzero(part_shi < 0)[0]

        if self.full_snapshot:
            if len(ind_not_in_sh) and not par['Input']['LoadFullFOF']:
                print("We have particles outside galaxies, but did not "
                      "load full FOF. Should not happen.")
                set_trace()

        part_shi[ind_not_in_sh] = self.nSH+1
        particles.galaxy = part_shi

        # Sort all particles in one go! Do *NOT* split
        # by type yet, because we first need them combined-by-radius for
        # vmax computation. Split by types will be done by 
        # external property calculation routine.

        if self.full_snapshot:
            particleSortList = np.lexsort(
                (particles.rad, particles.galaxy, part_fof))
        else:
            particleSortList = np.lexsort(
                (particles.rad, particles.galaxy))
            
        particles.ids = particles.ids[particleSortList]
        particles.mass = particles.mass[particleSortList]
        particles.pos = particles.pos[particleSortList, :]
        particles.vel = particles.vel[particleSortList, :]
        particles.type = particles.type[particleSortList]
        particles.rad = particles.rad[particleSortList]
        particles.galaxy = particles.galaxy[particleSortList]

        self.timeStamp.set_time('Sort particles')
        print("done (took {:.3f} sec.)." 
              .format(self.timeStamp.get_time()))

    def compute_galaxy_properties(self):          # Class: SnapshotOutput
        """Compute galaxy properties through external C code."""

        print("Computing galaxy properties... ", end = '', flush = True)

        particles = self.snap.particles

        # *********** IMPORTANT ********************************
        # This next line needs to be modified to point
        # to the full path of where the library has been copied.
        # *******************************************************

        ObjectFile = "/u/ybahe/ANALYSIS/PACKAGES/lib/galquant.so"

        numSH = self.sh_centreOfPotential.shape[0]
        numPart = particles.mass.shape[0]
        verbose = 0

        c_numPart = c.c_long(numPart)
        c_numSH = c.c_int(numSH)
        c_verbose = c.c_int(verbose)
        c_epsilon = c.c_float(self.snap.epsilon)

        mass_p = particles.mass.ctypes.data_as(c.c_void_p)
        pos_p = particles.pos.ctypes.data_as(c.c_void_p)
        vel_p = particles.vel.ctypes.data_as(c.c_void_p)
        type_p = particles.type.ctypes.data_as(c.c_void_p)
        shi_p = particles.galaxy.ctypes.data_as(c.c_void_p)
        rad_p = particles.rad.ctypes.data_as(c.c_void_p)
        ids_p = particles.ids.ctypes.data_as(c.c_void_p)

        cop_p = self.sh_centreOfPotential.ctypes.data_as(c.c_void_p)
        off_p = self.sh_offset.ctypes.data_as(c.c_void_p)
        len_p = self.sh_length.ctypes.data_as(c.c_void_p)
        lenType_p = self.sh_lengthType.ctypes.data_as(c.c_void_p)
        offType_p = self.sh_offsetType.ctypes.data_as(c.c_void_p)
        offTypeAp_p = self.sh_offsetTypeAp.ctypes.data_as(c.c_void_p)
        massTypeAp_p = self.sh_massTypeAp.ctypes.data_as(c.c_void_p)
        vmax_p = self.sh_vmax.ctypes.data_as(c.c_void_p)
        rvmax_p = self.sh_radOfVmax.ctypes.data_as(c.c_void_p)
        mtot_p = self.sh_mass.ctypes.data_as(c.c_void_p)
        comPos_p = self.sh_comPos.ctypes.data_as(c.c_void_p)
        zmfVel_p = self.sh_zmfVel.ctypes.data_as(c.c_void_p)
        rMax_p = self.sh_rMax.ctypes.data_as(c.c_void_p)
        rMaxType_p = self.sh_rMaxType.ctypes.data_as(c.c_void_p)
        comPosType_p = self.sh_comPosType.ctypes.data_as(c.c_void_p)
        zmfVelType_p = self.sh_zmfVelType.ctypes.data_as(c.c_void_p)
        velDisp_p = self.sh_velDisp.ctypes.data_as(c.c_void_p)
        angMom_p = self.sh_angMom.ctypes.data_as(c.c_void_p)
        axes_p = self.sh_axes.ctypes.data_as(c.c_void_p)
        axRat_p = self.sh_axRat.ctypes.data_as(c.c_void_p)
        kappaCo_p = self.sh_kappaCo.ctypes.data_as(c.c_void_p)
        smr_p = self.sh_smr.ctypes.data_as(c.c_void_p)
        
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

        self.timeStamp.set_time('Compute galaxy properties')               
        print("done (took {:.3f} sec.)." 
              .format(self.timeStamp.get_time()))

    def write(self, prefix=None):                  # Class: SnapshotOutput
        """
        Write internally-stored data to file.

        Parameters:
        -----------
        prefix : string, optional
            The HDF5 group name to which output will be written. If None
            (default), 'Snapshot_0xx/' will be used.
        """
        
        print("Writing output to HDF5 file... ", end = '')
        
        snap = self.snap
        isnap = snap.isnap      # For convenience
        galaxies = snap.target_galaxies
        particles = snap.particles

        # Set up snapshot-dependent prefix in output file:
        if prefix is None:
            snapPre = 'Snapshot_' + str(isnap).zfill(3) + '/'
        elif len(prefix) == 0:
            snapPre = ''
        elif prefix[-1] != '/':
            snapPre = prefix + '/'
        else:
            snapPre = prefix

        if len(snapPre) == 0:
            attDir = 'Header'
        else:
            attDir = snapPre

        # Write snapshot-specific header information:
        yb.write_hdf5_attribute(self.outloc, attDir, 'Redshift', snap.zred)
        yb.write_hdf5_attribute(self.outloc, attDir, 'aExp', 1/(1+snap.zred))
        yb.write_hdf5_attribute(self.outloc, attDir, 'NumSubhalo',
                                self.nSH)
        if self.full_snapshot:
            yb.write_hdf5_attribute(self.outloc, attDir, 'NumFOF', snap.nFOF)

        # Construct Cantor <--> subfind cross-indices: 
        if self.full_snapshot:
            sf_shi = self.gal_cantorID[snap.sh_galaxy]
            sh_sf = snap.shi[self.sh_galID] 

        # ------------------------------
        # i) Particle ID and radius list
        # ------------------------------

        yb.write_hdf5(particles.ids, self.outloc, 
                      snapPre + 'IDs', 
                      comment = "IDs of all particles associated with a "
                      "FOF group or subhalo (depending on settings). "
                      "Particles are sorted by FOF, then subhalo, then "
                      "type, and then radial distance from the subhalo "
                      "centre.")

        # If desired, write binding energies:
        if par['Output']['WriteBindingEnergy']:
            yb.write_hdf5(particles.binding_energy,
                          self.outloc, snapPre + 'BindingEnergy', 
                          comment = "Specific binding energy of all "
                          "particles in the "
                          "ID list. For particles not associated with a "
                          "subhalo, this is NaN, for others it is the "
                          "difference between KE and PE relative to their "
                          "subhalo (always negative). Note that it is also "
                          "NaN for centrals if central unbinding was "
                          "disabled. Units: km^2/s^2.")

        # If desired, write particle radii:
        if par['Output']['WriteRadii']:
            yb.write_hdf5(particles.rad, self.outloc,
                          snapPre + 'Radius',
                          comment = "Radial distance of each particle "
                          "from the centre-of-potential of its subhalo "
                          "(units: pMpc). Note that for subhaloes whose "
                          "centre-of-potential was taken from Subfind, "
                          "rounding errors may imply that no particle is "
                          "at a radius of exactly zero.")
        
        # --------------------------------------------------
        # ii) FOF offset/length list, and FOF --> SH indices
        # --------------------------------------------------
        
        if self.full_snapshot:
            yb.write_hdf5(self.fof_offset, self.outloc, snapPre+'FOF/Offset', 
                          comment = "Offset of FOF *index* i in ID list. "
                          "The particles belonging to this FOF are stored in "
                          "indices [offset]:[offset]+[length]. "
                          "Note that this includes a 'coda', so it has one "
                          "more elements than the number of FOFs, "
                          "and IDs may also be retrieved as "
                          "[offset[i]]:[offset[i+1]].")
            yb.write_hdf5(self.fof_length, self.outloc, snapPre+'FOF/Length', 
                          comment = 
                          'Number of particles in ID list that belong '
                          'to FOF *index* i. The particles belonging to this '
                          'FOF are stored in indices '
                          '[offset]:[offset]+[length]. '
                          'Note that subhaloes of each FOF are stored in '
                          'descending mass order, except for the central '
                          'subhalo, which always comes first.')

            # Write FOF --> subhalo links:
            yb.write_hdf5(self.fof_nsh, self.outloc, 
                          snapPre + 'FOF/NumOfSubhaloes',
                          comment = 'Number of subhaloes belonging to this '
                          'FOF.')
            yb.write_hdf5(self.fof_fsh, self.outloc, snapPre + 
                          'FOF/FirstSubhalo',
                          comment = 'Index of first subhalo belonging to this '
                          'FOF. Note that this will be >= 0 even if there is '
                          'not a single subhalo in the FOF group, to keep the '
                          'list monotonic. See CenSubhalo for a safe pointer '
                          'to the central subhalo, if it exists.')
            yb.write_hdf5(self.fof_cenSH, self.outloc, snapPre + 
                          'FOF/CenSubhalo', 
                          comment = 'Index of first subhalo belonging to this '
                          'FOF. -1 if the FOF has not a single subhalo.')
        
        # -----------------------------
        # iii) Subhalo index properties
        # -----------------------------

        grp = snapPre + 'Subhalo/'

        # Subhalo --> galaxy index
        yb.write_hdf5(self.sh_galID, self.outloc, grp + 'Galaxy', 
                      comment = "Galaxy ID for each subhalo index.")

        
        if self.full_snapshot:
            # Subhalo --> FOF
            yb.write_hdf5(self.sh_fof, self.outloc, grp + 'FOF_Index', 
                          comment = "Index of FOF group that this subhalo "
                          "belongs to.")
        
            # Cantor <--> Subfind subhalo cross indices
            yb.write_hdf5(sh_sf, self.outloc, 
                          grp + 'SubfindIndex', 
                          comment = "Index of the corresponding subhalo in "
                          "the Subfind catalogue. If this is < 0, it means "
                          "that the subhalo was lost by Subfind but recovered "
                          "by Cantor.")
            yb.write_hdf5(sf_shi, self.outloc, 
                          grp + 'IndexBySubfindID', 
                          comment = "Reverse index to identify the subhalo "
                          "corresponding to a given Subfind subhalo. To find "
                          "the (Cantor) subhalo corresponding to "
                          "Subfind subhalo index i, look at Cantor index "
                          "IndexBySubfindID[i].")

        # Subhalo --> particle indices
        yb.write_hdf5(self.sh_offset, self.outloc, 
                      grp + 'Offset', 
                      comment = "First index in ID list belonging to "
                      "subhalo i. The particles belonging to this subhalo "
                      "are stored in indices [offset]:[offset]+[length].")
        yb.write_hdf5(self.sh_length, self.outloc, 
                      grp + 'Length', 
                      comment = "Number of particles in ID list belonging "
                      "to subhalo i. The particles belonging to this subhalo "
                      "are stored in indices [offset]:[offset]+[length].")
        yb.write_hdf5(self.sh_offsetType, self.outloc, 
                      grp + 'OffsetType', 
                      comment = "First index in ID list belonging to "
                      "subhalo i (first index) and type j (second index). "
                      "The particles of this type belonging to this subhalo "
                      "are stored in indices [offset_i]:[offset_i+1]. "
                      "The last element (j = 6) serves as a coda for this.")
        yb.write_hdf5(self.sh_lengthType, self.outloc, 
                      grp + 'LengthType', 
                      comment = "Number of particles in ID list that belong "
                      "to subhalo i (first index) and have type j (second "
                      "index). These particles are stored at indices "
                      "[offset_i]:[offset_i+1].")
        
        if par['Output']['GalaxyProperties']:
            self.write_galaxy_properties(grp)


        self.timeStamp.set_time('Write Subhalo data')
        print("done (took {:.3f} sec.)." 
              .format(self.timeStamp.get_time()))
                
    def write_galaxy_properties(self, grp):
        """
        Write out galaxy properties, beyond segmentation map info.

        Parameters:
        -----------
        grp : str
            The HDF5 group to write data to.
        """

        # Maximum radius of subhalo particles
        yb.write_hdf5(self.sh_rMax, self.outloc, grp + 'MaxRadius',
                      comment = "Distance of furthest particle from subhalo "
                      "centre of potential (units: pMpc).")
        yb.write_hdf5(self.sh_rMaxType, self.outloc, grp + 'MaxRadiusType',
                      comment = "Distance of furthest particle of "
                      "a given type from subhalo centre of potential "
                      "(units: pMpc).")
        

        # ----------------------------------------------
        # iv) Main physical subhalo properties (for all)
        # ----------------------------------------------
        
        # Mass: total, and by type
        yb.write_hdf5(self.sh_mass, self.outloc, grp + 'Mass',
                      comment = "Total mass of each subhalo "
                      "(units: 10^10 M_Sun).")

        yb.write_hdf5(self.sh_massTypeAp[:, :, 4], self.outloc, 
                      grp + 'MassType', 
                      comment = "Mass per particle type of each subhalo "
                      "(units: 10^10 M_Sun).")

        # Subhalo coordinates
        yb.write_hdf5(self.sh_centreOfPotential, self.outloc,
                      grp + 'CentreOfPotential', 
                      comment = "Coordinates of particle with the lowest "
                      "gravitational potential (units: pMpc). For "
                      "galaxies that were not unbound directly "
                      "(i.e. centrals, if their unbinding was disabled), "
                      "the value is taken from the Subfind catalogue.")
        yb.write_hdf5(self.sh_monkPos, self.outloc,
                      grp + 'Position',
                      comment = "Coordinates of subhalo in final "
                      "unbinding iteration. Units: pMpc. NaN if particles "
                      "from this subhalo were not processed "
                      "(in particular possible for centrals).")
        yb.write_hdf5(self.sh_monkVel, self.outloc,
                      grp + 'Velocity',
                      comment = "Velocity of subhalo in final "
                      "unbinding iteration. Units: pMpc. NaN if particles "
                      "from this subhalo were not processed "
                      "(in particular possible for centrals).")
        yb.write_hdf5(self.sh_comPos, self.outloc,
                      grp + 'CentreOfMass', 
                      comment = "Coordinates of subhalo centre of mass "
                      "(units: pMpc).")
        yb.write_hdf5(self.sh_zmfVel, self.outloc,
                      grp + 'ZMF_Velocity', 
                      comment = "Velocity of the subhalo's zero-momentum "
                      "frame (i.e. mass-weighted velocity of all its "
                      "particles). Units: km/s.")

        # DM and stellar (total) velocity dispersion (index 2/5 => @infty)
        yb.write_hdf5(self.sh_velDisp[:, 2], self.outloc,
                      grp + 'VelocityDispersion_DM', 
                      comment = "Dark matter velocity dispersion of the "
                      "subhalo (units: km/s). A value of NaN indicates "
                      "that the subhalo has no DM, and 0 "
                      "(typically) means that it only has one DM "
                      "particle.")
        yb.write_hdf5(self.sh_velDisp[:, 5], self.outloc,
                      grp + 'VelocityDispersion_Stars', 
                      comment = "Stellar velocity dispersion of the "
                      "subhalo (units: km/s). A value of NaN indicates "
                      "that the subhalo has no stars, and 0 "
                      "(typically) means that it only has one stellar "
                      "particle.")
        
        # Gas, DM, stellar (total) angular momentum (index 2/5/8 => @infty)
        yb.write_hdf5(self.sh_angMom[:, 2, :], self.outloc,
                      grp + 'AngularMomentum_Gas', 
                      comment = "Angular momentum vector of gas particles "
                      "in the subhalo (units: 10^10 M_sun * pMpc * km/s). "
                      "The angular momentum is computed relative to the "
                      "subhalo centre of potential and the (total) gas "
                      "ZMF velocity.")
        yb.write_hdf5(self.sh_angMom[:, 5, :], self.outloc,
                      grp + 'AngularMomentum_DM', 
                      comment = "Angular momentum vector of DM particles "
                      "in the subhalo (units: 10^10 M_sun * pMpc * km/s). "
                      "The angular momentum is computed relative to the "
                      "subhalo centre of potential and the (total) DM "
                      "ZMF velocity.")
        yb.write_hdf5(self.sh_angMom[:, 8, :], self.outloc,
                      grp + 'AngularMomentum_Stars', 
                      comment = "Angular momentum vector of star particles "
                      "in the subhalo (units: 10^10 M_sun * pMpc * km/s). "
                      "The angular momentum is computed relative to the "
                      "subhalo centre of potential and the (total) stellar "
                      "ZMF velocity.")

        # Maximum circular velocity and its radius
        yb.write_hdf5(self.sh_vmax, self.outloc,
                      grp + 'Vmax', 
                      comment = "Maximum circular velocity of the "
                      "subhalo, calculated as max(sqrt(GM(<r)/r)). "
                      "Units: km/s.")
        yb.write_hdf5(self.sh_radOfVmax, self.outloc,
                      grp + 'RadiusOfVmax', 
                      comment = "Radius at which the circular velocity of "
                      "the subhalo is maximum, calculated as "
                      "argmax(sqrt(GM(<r)/r)). Units: pMpc.")

        # For clarity, 'extra' output (for massive galaxies) is outsourced:
        self.write_extra_output(grp)

                     
    def write_extra_output(self, snapPre):         #Class: SnapshotOutput
        """
        Write out extra physical properties of massive subhaloes.
            [Helper function of write()]

        These are all those that either have log_10 M_star/M_sun >= 8.5,
        or log_10 M_tot/M_sun >= 10.5. The rationale is that this excludes
        a gzillion tiny objects for which these values are pretty much
        meaningless anyway due to resolution limits.
        
        Parameters:
        -----------
        snapPre : string
            The snapshot-specific HDF5 group to which output is written.
            The extra output is written to [snapPre]/Extra/.
        """

        # Set up HDF5 directory to write to:
        grp = snapPre + 'Extra/'
        
        # --- Find galaxies worth the effort: ---------------
        # --- M_star or M_tot above specified threshold -----

        msub_min = par['Output']['Extra']['Mtot']
        mstar_min = par['Output']['Extra']['Mstar']
        if msub_min is None: 
            msub_min = 0
        if mstar_min is None:
            mstar_min = 0

        shi_extra = np.nonzero(
            (self.sh_mass >= 10.0**(msub_min-10)) |
            (self.sh_massTypeAp[:, 4, -1] >= 10.0**(mstar_min-10)))[0]

        # Also create reverse list, to find extraID from Subhalo ID
        extra_ids = yb.create_reverse_list(shi_extra, maxval = self.nSH-1)

        n_extra = len(shi_extra)
        print("Out of {:d} galaxies, {:d} are massive enough for extra "
              "output (={:.2f}%)." 
              .format(self.nSH, n_extra, n_extra/self.nSH*100))
        
        # ------------------------------------------
        # i) Indexing information extra <--> subhalo
        # ------------------------------------------

        yb.write_hdf5_attribute(self.outloc, grp, 'NumExtra', n_extra)

        yb.write_hdf5(shi_extra,
                      self.outloc, grp + 'SubhaloIndex',
                      comment = "Index into the `main' subhalo catalogue "
                      "for subhalo with extraID i (i.e. whose properties are "
                      "stored in index i in this extended catalogue).")
        yb.write_hdf5(extra_ids,
                      self.outloc, grp + 'ExtraIDs',
                      comment = "Index into this extended catalogue by "
                      "`main' subhalo index. Subhaloes that are not massive "
                      "enough to be included in the extended catalogue "
                      "have a value of -1.")

        # --------------------------------
        # ii) Particle indices by aperture
        # --------------------------------
        
        yb.write_hdf5(self.sh_offsetTypeAp[shi_extra, :, :],
                      self.outloc, grp + 'OffsetTypeApertures',
                      comment = "Index of first particle of subhalo i "
                      "(first index) and type j (second index) that is "
                      "more than 3/10/30/100 pkpc (third index) from "
                      "the centre of potential of its subhalo. To find "
                      "all particles of this type that are within one "
                      "of these apertures, load ids[offsetType:"
                      "offsetTypeApertures]. Note that this array has "
                      "no coda, so to load particles up to the outermost "
                      "one, the final particle index must be retrieved "
                      "from /Subhalo/OffsetType.")
        
        # --------------------------------------------------
        # iii) Properties that are the same for gas/DM/stars
        # --------------------------------------------------
        
        for ptype in [0, 1, 4, 5]:
            self.write_extra_type_properties(grp, ptype, shi_extra)

        # ----------------------------
        # iv) Star-specific properties
        # ----------------------------
        
        # Stellar mass percentiles:
        yb.write_hdf5(self.sh_smr[shi_extra, :, :], self.outloc,
                      grp + 'Stars/QuantileRadii', 
                      comment = "Radii containing {20, 50, 80} per cent "
                      "(3rd index) of the stellar mass within "
                      "{30 pkpc, infty} (2nd index) from the subhalo centre "
                      "(units: pMpc). "
                      "The radii are interpolated between that "
                      "of the outermost particle enclosing less than "
                      "the target mass and the one "
                      "immediately beyond it. If there is only one "
                      "star particle, the result is taken "
                      "as half its radius.")
 
        # Kinematic morphology parameter:
        yb.write_hdf5(self.sh_kappaCo[shi_extra], self.outloc,
                      grp + 'Stars/KappaCo',
                      comment = "Stellar kinematic morphology parameter "
                      "as in Correa+17 (based on Sales+10); defined as "
                      "K_rot/K_tot with "
                      "K_rot = sum(1/2 * m_i * v'_i**2) and "
                      "K_tot = sum(1/2 * m_i * v_i**2), where "
                      "v'_i = L_z_i / (m_i * R_i), with L_z_i the (ith "
                      "particle's) component of angular momentum along the "
                      "total angular momentum axis and "
                      "R_i its perpendicular distance from the axis. " 
                      "Only particles with positive-definite L_z are "
                      "considered. Particles are selected within "
                      "{10, 30} pkpc (2nd index), and both the reference "
                      "velocity and angular momentum are computed over "
                      "all (star) particles in the same aperture. The "
                      "reference position is the subhalo "
                      "centre of potential. The corresponding "
                      "angular momentum axes are stored in "
                      "'Extra/AngularMomentum'.")

                                                  # Class: SnapshotOutput
    def write_extra_type_properties(self, grp, ptype, shi_extra):
        """
        Write extra properties that are the same for several types.
           [Helper function of write_extra_output()]

        Parameters:
        -----------
        grp : string
            The HDF5 group to which the extra output is written 
            (within current snapshot group).
        ptype : int
            The particle type code for which to write output
            (0=gas, 1=DM, 4=stars, 5=BHs)
        shi_extra : ndarray (int)
            The indices of subhaloes that are massive enough to warrant
            writing the extended output.
        """ 

        typeNames = ['Gas', 'DM', '', '', 'Stars', 'BHs']
        typeNamesLCS = ['gas', 'DM', '', '', 'star', 'BH']
        typeNamesLCP = ['gas', 'DM', '', '', 'stars', 'BHs']

        # Offsets into the ptype+aperture output arrays 
        zmfOff = np.array([0, 3, 6, 6, 6, 9, 9])  # ZMF-vel & ang.mom.
        axOff = np.array([0, 0, 3, 3, 3, 6, 6])   # Vel. disp. & MIT

        grp_type = grp + typeNames[ptype] + '/'
        tns = typeNamesLCS[ptype]
        tnp = typeNamesLCP[ptype]

        # Mass by aperture:
        yb.write_hdf5(self.sh_massTypeAp[shi_extra, ptype, :4], self.outloc,
                      grp_type + 'ApertureMasses', 
                      comment = "Sum of " + tns + " particles masses within "
                      "{3, 10, 30, 100} pkpc from the subhalo centre of "
                      "potential. Units: 10^10 M_Sun.")

        # Centre of mass:
        yb.write_hdf5(self.sh_comPosType[shi_extra, ptype, :], self.outloc,
                      grp_type + 'CentreOfMass', 
                      comment = "Centre of mass of " + tns + " particles " 
                      "in this subhalo (units: pMpc).")

        if ptype == 5: return

        # ZMF velocity (only write 30pkpc -- offset+1)
        yb.write_hdf5(self.sh_zmfVelType[shi_extra, zmfOff[ptype]+1, :], 
                      self.outloc, grp_type + 'ZMF_Velocity_30kpc', 
                      comment = "Zero-momentum-frame velocity of " + tns + 
                      "particles within 30 pkpc from the subhalo centre "
                      "of potential (units: km/s).")

        # Angular momentum vector:
        # (note that infty is already written as standard output)
        yb.write_hdf5(
            self.sh_angMom[shi_extra, zmfOff[ptype]:zmfOff[ptype+1]-1, :], 
            self.outloc, grp_type + 'AngularMomentum', 
            comment = "Angular momentum vector of " + tns + " particles "
            "within {10, 30} pkpc from the subhalo centre "
            "of potential (units: 10^10 M_sun * pMpc * km/s). "
            "The angular momentum is computed relative to the "
            "subhalo centre of potential and the respective "
            "particles' ZMF velocity.")
        
        if (ptype == 1 or ptype == 4):
            # Moment-of-inertia tensor axes and ratios:
            yb.write_hdf5(
                self.sh_axes[shi_extra, axOff[ptype]:axOff[ptype+1], :, :], 
                self.outloc, grp_type + 'Axes', 
                comment = "Principal axes of the " + tns + 
                " moment-of-inertia tensor, within {10, 30, infty} pkpc "
                "(2nd index) from the subhalo centre of potential. "
                "The 3rd index specifies the minor (0), intermediate (1), " 
                "and major (2) axis; the 4th index specifies the x/y/z "
                "component of the (unit) vector along the respective axis."
                "A value of NaN indicates that there are no particles "
                "within the respective aperture. The corresponding axis "
                "ratios are stored in 'Extra/AxisRatios'.")
            
            yb.write_hdf5(
                self.sh_axRat[shi_extra, axOff[ptype]:axOff[ptype+1], :],
                self.outloc, grp_type + 'AxisRatios', 
                comment = "Ratio of the minor and intermediate "
                "axis to the major axis, respectively (0 and 1 along 3rd "
                "index) of the " + tns + " moment-of-inertia tensor "
                "within {10, 30, infty} pkpc (2nd index)."
                "A value of NaN indicates that there are no particles "
                "within the respective aperture. The corresponding axis "
                "(unit) vectors are stored in 'Extra/Axes'.");

            # Velocity dispersions
            # (infty value already written as main output)
            yb.write_hdf5(
                self.sh_velDisp[shi_extra, axOff[ptype]:axOff[ptype+1]-1], 
                self.outloc, grp_type + 'VelocityDispersion', 
                comment = "Velocity dispersion of " + tns + " particles "
                "within {10, 30} pkpc from the subhalo centre "
                "of potential (units: km/s). The reference velocity is "
                "taken as the ZMF velocity of " + tns + " particles within "
                "the same aperture, and particles are weighted by mass.")