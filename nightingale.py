"""Main script of NIGHTINGALE.

Started 17 Feb 2025.
"""

import numpy as np
import tools
import timestamp as ts
import params
from snapshot import Snapshot



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

    # Set up a 'Snapshot' class instance that holds its basic properties:
    targetSnap = Snapshot(par)
    priorSnap = Snapshot(par, offset=-1)
    prePriorSnap = Snapshot(par, offset=-2)

    # Load target snapshot subhaloes
    subhaloes = SnapshotGalaxies(targetSnap)

    # Load information about subhaloes in prior snapshots
    # TO BE ADDED once this has become clearer

    # Load particle data
    particles = SnapshotParticles(targetSnap)

    # --------------------------------------------------------------------

    # Initialise particle re-assignment
    particles.initialise_subhaloes()

    # Main loop over galaxies to assemble, unbind, and assign particles.
    for ish in range(subhaloes.n_subhaloes):

        # Initialise the galaxy with basic information
        galaxy = TargetGalaxy(subhaloes, ish)

        # Find the source particles of the galaxy (separate from
        # initialisation to allow for easier swapping)
        galaxy_particles = galaxy.find_source_particles()

        # Perform gravitational unbinding
        final_subhalo_coords = galaxy_particles.unbind()
        subhaloes.update_coordinates(ish, final_subhalo_coords)

        # Update full particle membership
        particles.update_membership(galaxy_particles)


    # All galaxies are processed now. Resolve assignment conflicts
    particles.resolve_assignment_conflicts()

    # ----------------------------------------------------------------------

    # We are done with the main part now -- all particles are assigned to their
    # final subhalo. Hand over to output processing...

    output = Output(par, isnap)
    output.load_particle_data(subhaloes, particles)

    # Compute any quantities beyond pure subhalo-->particle assignments
    output.compute_secondary_quantities()

    # Write output
    output.write()



if __name__ == "__main__":
    main()
    print("Done!")
