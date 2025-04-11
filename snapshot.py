"""Snapshot class to hold basic snapshot information.

Started 17 Feb 2025.
"""

import numpy as np
import ioi
import ion
import cosmo
from pdb import set_trace

class Snapshot:
    """Hold and process snapshot-level information."""  

    def __init__(self, sim, offset=0):
        self.sim = sim
        self.par = sim.par
        self.verbose = self.par['Verbose']

        self.isnap = self.par['Sim']['Snapshot'] + offset
        if self.isnap < 0:
            return

        self.offset = offset

        self.redshift = sim.redshifts[self.isnap]
        self.aexp = sim.aexps[self.isnap]

        self.epsilon = cosmo.compute_softening_length(self.redshift)

        # Compute Hubble parameter for this snapshot.
        self.hubble_z = cosmo.compute_hubble_z(self.redshift)
        if self.par['Check']['NoHubble']:
            self.hubble_z = 0

        # Set input file names
        self.set_input_file_names()

        print(f"Finished initialization of snapshot {self.isnap}.")
        print(f"Redshift = {self.redshift:.2f}, H(z) = {self.hubble_z:.2f} "
              f"km/s/Mpc")
        
    def set_input_file_names(self):
        """Form all input file names relevant to this snapshot."""
        isnap = self.isnap
        par = self.par

        self.snapshot_file = ioi.form_snapshot_file(par, isnap)
        if self.par['Input']['FromNightingale'] and self.offset < 0:
            self.subhalo_file = (
                ion.form_nightingale_property_file(self.par, self.isnap))
            self.subhalo_particle_file = (
                ion.form_nightingale_id_file(self.par, self.isnap))
        else:
            self.subhalo_file = ioi.form_subhalo_file(par, isnap)
            self.subhalo_particle_file = ioi.form_subhalo_particle_file(par, isnap)
        #self.subhalo_membership_file = ioi.form_subhalo_membership_file(
        #    par, isnap)
        
        self.nightingale_property_file = ion.form_nightingale_property_file(
            par, isnap)
        self.nightingale_id_file = ion.form_nightingale_id_file(par, isnap)
        self.nightingale_waitlist_file = ion.form_nightingale_waitlist_file(
            par, isnap)

        if self.par['Verbose']:
            print(f"Set file names for snapshot {isnap}:")
            print(f"Snapshot file: '{self.snapshot_file}'")
            print(f"Subhalo file: '{self.subhalo_file}'")
            print(f"Subhalo particle file: '{self.subhalo_particle_file}'")
            print(f"Nightingale catalogue: '{self.nightingale_property_file}'")
            print(f"Nightingale ID file: '{self.nightingale_id_file}'")
