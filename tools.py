"""Tool collection for nightingale.

Started 17 Feb 2025.
"""

import hdf5
import os
import subprocess
import re
import numpy as np
from pdb import set_trace

def dict2att(dictIn, outloc, container='Header', pre='',
             bool_as_int=True):
    """
    Write all elements of a dictionary as attributes to an HDF5 file.

    Typically, this is used to write the paramter structure of a 
    program to its output. If keys are themselves dictionaries, these 
    are recursively output with an underscore between them 
    (e.g. dictIn['Sim']['Input']['Flag'] --> 'Sim_Input_Flag').

    Parameters:
    -----------
    dictIn : dict
        The dictionary to output.
    outloc : string
        The HDF5 file name to write the dictionary to.
    container : string, optional
        The container to which the dict's elements will be written as 
        attributes (can be group or dataset). The default is a group 
        'Header'. If the container does not exist, a group with the specified
        name will be implicitly created.
    pre : string, optional:
        A prefix given to all keys from this dictionary. This is mostly
        used to output nested dictionaries (see description at top), but 
        may also be used to append a 'global' prefix to all keys.
    bool_as_int : bool, optional:
        If True (default), boolean keys will be written as 0 or 1, instead
        of as True and False.
    """

    if len(pre):
        preOut = pre + '_'
    else:
        preOut = pre

    for key in dictIn.keys():

        value = dictIn[key]
        if isinstance(value, dict):
            # Nested dict: call function again to iterate
            dict2att(value, outloc, container = container, 
                     pre = preOut + key, bool_as_int=bool_as_int)
        else:
            # Single value: write to HDF5 file

            if value is None:
                value = 0

            if bool_as_int and isinstance(value, bool):
                value = int(value)

            if isinstance(value, str):
                value = np.string_(value)

            hdf5.write_hdf5_attribute(
                outloc, container, preOut + key, value)


def dict2out(dictIn, bool_as_int=True, pre=''):
    """
    Write all elements of a dictionary as attributes to the output.

    If any keys are themselves dictionaries, these 
    are recursively output with an underscore between them 
    (e.g. dictIn['Sim']['Input']['Flag'] --> 'Sim_Input_Flag').

    Parameters:
    -----------
    dictIn : dict
        The dictionary to output.
    bool_as_int : bool, optional:
        If True (default), boolean keys will be written as 0 or 1, instead
        of as True and False.
    """

    if len(pre):
        preOut = pre + '_'
    else:
        preOut = pre

    for key in dictIn.keys():

        value = dictIn[key]
        if isinstance(value, dict):
            # Nested dict: call function again to iterate
            dict2out(value, pre = preOut + key, bool_as_int=bool_as_int)
        else:
            # Single value: write to HDF5 file

            if value is None:
                value = 0

            if bool_as_int and isinstance(value, bool):
                value = int(value)

            print(preOut + key, ': ', value) 


def eprint(string, linestyle = '-', padWidth = 1, lineWidth = 1, 
           textPad = None):
    """
    Print a string with padding.

    Parameters:
    -----------

    string : string
        The text string to print in the middle of the padding.
    linestyle : string, optional
        The pattern to use for framing the string, default: '-'.
    padWidth : int, optional
        The number of blank lines to print either side of the frame,
        defaults to 1.
    lineWidth : int, optional
        The number of lines to draw either side of the string,
        defaults to 1.
    textPad : int, optional
        If not None, the width to which the string is padded with 
        the framing linestyle (centrally aligned).
    """
    
    stringLength = len(string)
    lineElementLength = len(linestyle)

    if textPad is not None:
        padLength = textPad//2 - (stringLength//2 + 1)
        if padLength < 0:
            padLength = 0
        numPad = padLength//lineElementLength + 1
        padString = (linestyle * numPad)[:padLength]

        string = padString + ' ' + string + ' ' + padString
        string = string[:textPad]
        stringLength = len(string)

    # Work out how often the line element must be repeated:
    numLines = stringLength // lineElementLength
    line = (linestyle * numLines)[:stringLength]
        
    for ipad in range(padWidth):
        print("")
    for iline in range(lineWidth):
        print(line)
    print(string)
    for iline in range(lineWidth):
        print(line)
    for ipad in range(padWidth):
        print("")


def print_memory_usage(pre=''):
    """Dummy that does nothing."""
    pass

def print_memory_usage_real(pre=''):
    """
    Print (total) current memory usage of the program, as seen by the OS.

    Parameters:
    -----------
    pre : str, optional
        A string that precedes the memory usage message (typically used
        to indicate where in the program the measurement is done).
    """

    # Get program's PID
    pid = os.getpid()

    # Call pmap to get memory footprint
    ret = subprocess.check_output(['pmap', str(pid)])

    # Extract the one number we care about from the output: total usage
    rets = str(ret)
    ind = rets.find("Total:")
    part = rets[ind:]
    colInd = part.find(":")
    kInd = part.find("K")
    mem = int(part[colInd+1 : kInd])

    print(pre + "Using {:d} KB (={:.1f} MB, {:.1f} GB)." 
          .format(mem, mem/1024, mem/1024/1024))


class SplitList:
    """Class to simplify particle lookup by a given property."""

    def __init__(self, quant, lims):
        """
        Class constructur.

        Parameters:
        -----------
        quant : ndarray
            The quantity by which elements should be retrievable.
        lims : ndarray
            The boundaries (in the same quantity and units as quant) of
            each retrievable 'element'.
        """

        self.argsort = np.argsort(quant)
        self.splits = np.searchsorted(quant, lims, sorter = self.argsort)

    def __call__(self, index):
        """
        Return all input elements that fall in a given bin.

        Parameters:
        -----------
        index : int
            The index corresponding to the supplied 'lims' array for which
            elements should be retrieved. All elements with 'quant' between
            lims[index] and lims[index+1] will be returned.

        Returns:
        --------
        elements : ndarray
            The elements that lie in the desired quantity range.
        """

        return self.argsort[self.splits[index]:self.splits[index+1]]

def key_to_attribute_name(key):
    """Convert the CamelCase name of a key to snake_case for attribute use.

    Taken from https://stackoverflow.com/questions/1175208.
    """
    if 'IDs' in key:
        key = key.replace('IDs', 'Ids')

    pattern = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
    name = pattern.sub('_', key).lower()

    print(f"Converted '{key}' to '{name}'")
    return name


def periodic_wrapping(r, boxsize, return_copy=False, mode='centre'):
    """
    Apply periodic wrapping to an input set of coordinates. 
    
    Parameters
    ----------
    r : ndarray(float) [N, 3] 
        The coordinates to wrap.
    boxsize : float                                               
        The box size to wrap the coordinates to. The units must correspond to
        those used for `r`. 
    return_copy : bool, optional
        Switch to return a (modified) copy of the input array, rather than
        modifying the input in place (which is the default).      
    mode : string, optional                    
        Specify whether coordinates should be wrapped to within -0.5 --> 0.5
        times the boxsize ('centre', default) or 0 --> 1 boxsize ('corner').
    
    Returns
    -------
    r_wrapped : ndarray(float) [N, 3]
        The wrapped coordinates. Only returned if `return_copy` is True,    
        otherwise the input array `r` is modified in-place.
    """
    if mode in ['centre', 'center']:
        shift = 0.5 * boxsize
    elif mode in ['corner']:
        shift = 0.0
    else:
        raise ValueError(f'Invalid wrapping mode "{mode}"!')
    
    if return_copy:
        r_wrapped = ((r + shift) % boxsize - shift)
        return r_wrapped

    # To perform the wrapping in-place, break it down into three steps
    r += shift
    r %= boxsize
    r -= shift
    
