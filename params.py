"""Input parameter handling.

Started 17 Feb 2025.
"""

import yaml
import argparse
from tools import eprint, dict2out
from pdb import set_trace
import numpy as np

def parse_arguments():
    """Parse the command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run Nightingale to post-process a snapshot."
    )

    # Parameter file is non-optional    
    parser.add_argument(
        'param_file',
        help="Parameter file for this run."
    )

    parser.add_argument(
        '-s', '--snapshot', type=int,
        help="The snapshot index to process. If given, this overrides any "
             "value in the parameter file."
    )
    
    args = parser.parse_args()

    # --------------------------------
    # Consistency checks to go here...
    # --------------------------------

    return args


def parse_parameter_file(par_file):
    """Parse the YAML input parameter file to `par' dict."""

    with open(par_file, 'r') as stream:
        try:
            par = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return par


def override_parameters(par, args):
    # Check if simulation was specified on the command line:

    if 'snapshot' in args:
        if args.snapshot is not None:
            par['Sim']['Snapshot'] = args.snapshot



def setup(par):
    """Set derived parameters and do consistency checks."""

    # Form snapshot file:
    snap_str = f'{par['Sim']['Snapshot']:04d}'
    #par['Sim']['SnapshotFile'] = (
    #    par['Sim']['SnapshotFile'].replace('XXX', snap_str))

    # Set derived parameters:
    if (par['Lost']['FindTemporarilyLost'] or 
        par['Lost']['FindPermanentlyLost']):
        par['Lost']['Recover'] = True
    else:
        par['Lost']['Recover'] = False

    if par['Input']['IncludeBoundary']:
        par['Input']['TypeList'] = np.arange(6, dtype = np.int8)
    else:
        par['Input']['TypeList'] = np.array([0, 1, 4, 5])

    # Consistency checks and warnings:
    #if par['Lost']['FindPermanentlyLost'] and not par['Input']['FromCantor']:
    #    raise Exception("Unbinding lost galaxies requires "
    #                    "loading CANTOR input.")
    
    if par['Sources']['Sats'] or par['Sources']['FOF']:
        if not par['Input']['RegularizedCens']:
            eprint("WARNING", textPad = 60, linestyle = '=')
            print("Including root sat/fof particles without using regularized")
            print("cen-sat input may lead to nonsensical results.")
            print("")
            
        if not par['Input']['FromNightingale']:
            eprint("WARNING", textPad = 60, linestyle = '=')
            print("Including root sat/fof particles without re-using Nightingale")
            print("output may lead to nonsensical results.")
            print("")

    if (not par['Output']['COPAtUnbinding'] 
        or par['Output']['WriteBindingEnergy']):
        par['Monk']['ReturnBindingEnergy'] = 1
    else:
        par['Monk']['ReturnBindingEnergy'] = 0

    if par['Sources']['Prior'] and not par['Input']['FromNightingale']:
        eprint("INCONSISTENCY", textPad = 60, linestyle = '@')
        #raise Exception("Cannot use previous snapshot without "
        #                "loading Nightingale output.")

    if par['Lost']['MaxLostSnaps'] is None:
        par['Lost']['MaxLostSnaps'] = np.inf

    if (par['Lost']['Recover'] and par['Sources']['Subfind'] 
        and not par['Sources']['Prior']):
        eprint("WARNING", textPad = 60, linestyle = '=')
        print("Included subfind particles and included lost")
        print("galaxies, but not loaded previous Cantor particles.")

    # Write parameter structure to output:
    print("\n----------------------------------------------------------")
    print("Nightingale configuration:")
    print("-----------------------------------------------------------")
    dict2out(par)
    print("-----------------------------------------------------------")
    print("")

    return

