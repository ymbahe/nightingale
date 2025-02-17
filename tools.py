"""Tool collection for nightingale.

Started 17 Feb 2025.
"""

import hdf5
import os
import subprocess

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