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

    # Set derived parameters:
    if par['Input']['IncludeBoundary']:
        par['Input']['TypeList'] = np.arange(6, dtype = np.int8)
    else:
        par['Input']['TypeList'] = np.array([0, 1, 4, 5])

    if (not par['Output']['COPAtUnbinding']):
        par['Monk']['ReturnBindingEnergy'] = 1
    else:
        par['Monk']['ReturnBindingEnergy'] = 0

    if par['Sources']['Prior'] and not par['Input']['FromNightingale']:
        eprint("INCONSISTENCY", textPad = 60, linestyle = '@')
        #raise Exception("Cannot use previous snapshot without "
        #                "loading Nightingale output.")

    # Write parameter structure to output:
    print("\n----------------------------------------------------------")
    print("Nightingale configuration:")
    print("-----------------------------------------------------------")
    dict2out(par)
    print("-----------------------------------------------------------")
    print("")

    return

