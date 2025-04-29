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
from ion import Output

#import gc

from pdb import set_trace

#gc.disable()

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
    sim.priorSnap.subhaloes = SnapshotGalaxies(sim.priorSnap, kind='prior')
    sim.prePriorSnap.subhaloes = SnapshotGalaxies(
        sim.prePriorSnap, kind='prior')

    # Load particle data
    particles = SnapshotParticles(targetSnap)

    # --------------------------------------------------------------------

    # Initialise particle re-assignment: set all relevant particles to central
    particles.initialise_memberships()
    subhaloes.initialise_new_coordinates()

    # If we want to unbind centrals WITH satellites still attached, that
    # needs to happen here...

    # Main loop over galaxies to assemble, unbind, and assign particles.
    # We process subhaloes in inverse order of depth.
    for idepth in range(np.max(subhaloes.depth), 0, -1):

        if idepth != 1: continue

        for ish in range(subhaloes.n_input_subhaloes):
            
            # Only process the subhalo in this turn if its depth is the one
            # that is currently being analysed
            if subhaloes.depth[ish] != idepth:
                continue

            #gc.collect()
            #if idepth == 1 and ish == 4000: set_trace()
            #if idepth == 1 and ish >= 5000: break
            #if ish < 4000: continue
            #if idepth == 1 and ish >= 10000: gc.collect()
            
            # Don't bother with 'fake' subhaloes
            if subhaloes.number_of_bound_particles[ish] <= 0:
                continue
            print(f"Subhalo {ish} -- Depth = {subhaloes.depth[ish]}, "
                  f"N_bound_input = "
                  f"{subhaloes.number_of_bound_particles[ish]}"
            )
        
            # Initialise the galaxy with basic information
            galaxy = TargetGalaxy(subhaloes, ish)

            # Find the source particles of the galaxy (separate from
            # initialisation to allow for easier swapping)
            galaxy_particles = galaxy.find_source_particles()
            if galaxy_particles is None:
                continue

            # Perform gravitational unbinding
            final_subhalo_coords, m_bound = galaxy_particles.unbind()
            #subhaloes.register_unbinding_result(
            #    ish, final_subhalo_coords, m_bound)
            
            # Update full particle membership
            #particles.update_membership(galaxy_particles, galaxy)

    # Filter out 'waitlist' particles
    #particles.filter_out_waitlist()

    # All subhaloes are processed now, and `particles` contains their final
    # membership information. Update subhalo coordinates in the catalogue.
    #subhaloes.update_coordinates()

    # Get rid of any particles that are not bound
    #particles.reject_unbound()


    # ----------------------------------------------------------------------

    # Almost done -- hand over to output handling...
    #output = Output(par, targetSnap, subhaloes, particles)

    # Trim the particle list to only include those in identified subhaloes
    #particles.switch_memberships_to_output(output.output_shi_from_input_shi)

    # Compute the core subhalo properties: particle membership and position
    #output.prepare()

    # Compute what we want to know
    #output.compute_output_quantities()

    # Write output
    #output.write()


if __name__ == "__main__":
    main()
    print("Done!")
