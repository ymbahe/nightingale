"""Main script of NIGHTINGALE.

Started 17 Feb 2025.
"""

import numpy as np
import tools
import timestamp as ts
import params
from snapshot import Snapshot
from simulation import Simulation
from galaxies import SnapshotGalaxies, TargetGalaxy
from particles import SnapshotParticles

def main():
    """Main function of Nightingale, processes one snapshot.

    Parses the parameter file and command-line options, reads data,
    processes individual subhaloes, gathers results, derives output,
    writes output.
    """

    # Set up top-level time stamp:
    timeStamp = ts.TimeStamp()
    tools.print_memory_usage("At start:")

    # Parse command-line options:
    args = params.parse_arguments()

    # Parse input parameter file:
    par = params.parse_parameter_file(args.param_file)

    # Check whether any parameters in the param file should be overriden by
    # command-line arguments
    params.override_parameters(par, args)

    # Set derived parameters and perform consistency checks:
    params.setup(par)

    # -------------------------------------------------------------------

    # Set up a 'Simulation' class instance for simulation-wide properties
    sim = Simulation(par)

    # Set up a 'Snapshot' class instance that holds its basic properties:
    targetSnap = Snapshot(sim)
    sim.targetSnap = targetSnap
    sim.priorSnap = Snapshot(sim, offset=-1)
    sim.prePriorSnap = Snapshot(sim, offset=-2)

    # Load target snapshot subhaloes
    subhaloes = SnapshotGalaxies(targetSnap)
    targetSnap.subhaloes = subhaloes

    # Load information about subhaloes in prior snapshots
    priorSnap.subhaloes = SnapshotGalaxies(sim.priorSnap, kind='prior')
    prePriorSnap.subhaloes = SnapshotGalaxies(sim.prePriorSnap, kind='prior')

    # Load particle data
    particles = SnapshotParticles(targetSnap)

    # --------------------------------------------------------------------

    # Initialise particle re-assignment: set all relevant particles to central
    particles.initialise_memberships()
    subhaloes.initialise_new_coordinates()

    # Main loop over galaxies to assemble, unbind, and assign particles.
    for ish in range(subhaloes.n_subhaloes):

        # Initialise the galaxy with basic information
        galaxy = TargetGalaxy(subhaloes, ish)

        # Find the source particles of the galaxy (separate from
        # initialisation to allow for easier swapping)
        galaxy_particles = galaxy.find_source_particles()

        # Perform gravitational unbinding
        final_subhalo_coords = galaxy_particles.unbind()
        subhaloes.register_new_coordinates(ish, final_subhalo_coords)

        # Update full particle membership
        particles.update_membership(galaxy_particles)

    # All subhaloes are processed now, and `particles` contains their final
    # membership information. Update subhalo coordinates in the catalogue.
    subhaloes.update_coordinates()

    # ----------------------------------------------------------------------

    # Almost done -- hand over to output handling...
    output = Output(par, targetSnap, subhaloes, particles)

    # Process the particle-subhalo links for output
    output.process_subhaloes_and_membership()

    # Compute any quantities beyond pure subhalo-->particle assignments
    output.compute_secondary_quantities()

    # Write output
    output.write()



if __name__ == "__main__":
    main()
    print("Done!")
