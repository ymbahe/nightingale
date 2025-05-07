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
import ion

from pdb import set_trace

def main():
    """Main function of Nightingale, processes one snapshot.

    Parses the parameter file and command-line options, reads data,
    processes individual subhaloes, gathers results, derives output,
    writes output.
    """

    # Set up top-level time stamp:
    timer = ts.TimeStamp()
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

    timer.set_time('Initialization')
    print(f"Initialization done, took {timer.get_time():.2f} sec.")
    
    # -------------------------------------------------------------------

    # Set up a 'Simulation' class instance for simulation-wide properties
    sim = Simulation(par)

    # Set up a 'Snapshot' class instance that holds its basic properties:
    targetSnap = Snapshot(sim)
    sim.targetSnap = targetSnap
    sim.priorSnap = Snapshot(sim, offset=-1)
    sim.prePriorSnap = Snapshot(sim, offset=-2)
    timer.set_time('Snapshot initialization')
    
    # Load target snapshot subhaloes
    print(f"Load subhaloes... ", flush=True)
    subhaloes = SnapshotGalaxies(targetSnap)
    targetSnap.subhaloes = subhaloes

    # Load information about subhaloes in prior snapshots
    sim.priorSnap.subhaloes = SnapshotGalaxies(sim.priorSnap, kind='prior')
    sim.prePriorSnap.subhaloes = SnapshotGalaxies(
        sim.prePriorSnap, kind='prior')
    timer.set_time('Load subhaloes')
    print(f"   ... done loading subhaloes [{timer.get_time():.2f} sec.]", flush=True)
    
    # Load particle data
    print(f"Load particle data...")
    particles = SnapshotParticles(targetSnap)
    timer.set_time('Load particles')
    print(f"   ... done loading particles [{timer.get_time():.2f} sec.]", flush=True)

    # --------------------------------------------------------------------

    # Initialise particle re-assignment: set all relevant particles to central
    particles.initialise_memberships()
    subhaloes.initialise_new_coordinates()
    timer.set_time('Initialize particles')
    
    # If we want to unbind centrals WITH satellites still attached, that
    # needs to happen here...

    # Main loop over galaxies to assemble, unbind, and assign particles.
    # We process subhaloes in inverse order of depth.

    timedata = {}
    timedata['Time'] = np.zeros(subhaloes.n_input_subhaloes)
    timedata['NPart'] = np.zeros(subhaloes.n_input_subhaloes, dtype=int)
    timedata['NMassive'] = np.zeros(subhaloes.n_input_subhaloes, dtype=int)
    
    for idepth in range(np.max(subhaloes.depth), 0, -1):
        ii = 0
        for ish in range(subhaloes.n_input_subhaloes):

            # Only process the subhalo in this turn if its depth is the one
            # that is currently being analysed
            if subhaloes.depth[ish] != idepth:
                continue
            # Don't bother with 'fake' subhaloes
            if subhaloes.number_of_bound_particles[ish] <= 0:
                continue
            ii += 1
            if verbose > 0 or ii % 1000 == 0:
                print(f"\nSubhalo {ish} -- Depth = {subhaloes.depth[ish]}, "
                      f"N_bound_input = "
                      f"{subhaloes.number_of_bound_particles[ish]}"
                )

            sub_timer = ts.TimeStamp()
            
            # Initialise the galaxy with basic information
            galaxy = TargetGalaxy(subhaloes, ish)
            
            # Find the source particles of the galaxy (separate from
            # initialisation to allow for easier swapping)
            if verbose > 0:
                print(f"Finding source particles of subhalo {ish}...")
            galaxy_particles = galaxy.find_source_particles()
            if galaxy_particles is None:
                continue
            sub_timer.set_time('Source finding')
            if verbose > 0:
                print(f"   ... done [{sub_timer.get_time():.2f} sec.]")
            
            # Perform gravitational unbinding
            if verbose > 0:
                print(f"Unbinding source particles for subhalo {ish}...")
            final_subhalo_coords, n_massive, m_bound, m_passive = (
                galaxy_particles.unbind())
            sub_timer.set_time('Unbinding')
            if verbose > 0:
                print(f"   ... done [{sub_timer.get_time():.2f} sec.]")

            subhaloes.register_unbinding_result(
                ish, final_subhalo_coords, m_bound, m_passive)
            timedata['Time'][ish] = sub_timer.get_time()
            timedata['NPart'][ish] = galaxy_particles.num_part
            timedata['NMassive'][ish] = n_massive
            
            # Update full particle membership
            particles.update_membership(galaxy_particles, galaxy)
            sub_timer.set_time('Registration')
            timer.copy_times(sub_timer)
            if verbose > 0:
                sub_timer.print_time_usage(
                    f'Finished subhalo {ish}', caption_style='sec')

    ion.write_timedata(timedata, par, targetSnap.isnap)
            
    # Filter out 'waitlist' particles
    timer.start_time()
    particles.filter_out_waitlist()
    timer.set_time('Filter out waitlist')
    
    # All subhaloes are processed now, and `particles` contains their final
    # membership information. Update subhalo coordinates in the catalogue.
    subhaloes.update_coordinates()

    # Get rid of any particles that are not bound
    particles.reject_unbound()
    timer.set_time('Final particles processing')

    # ----------------------------------------------------------------------

    # Almost done -- hand over to output handling...
    output = Output(par, targetSnap, subhaloes, particles)

    # Trim the particle list to only include those in identified subhaloes
    particles.switch_memberships_to_output(output.output_shi_from_input_shi)

    # Compute the core subhalo properties: particle membership and position
    output.prepare()

    # Compute what we want to know
    output.compute_output_quantities()

    # Write output
    output.write()
    timer.set_time('Output writing')

    timer.print_time_usage(f"Finished snapshot {targetSnap.isnap}")


if __name__ == "__main__":
    main()
    print("Done!")
