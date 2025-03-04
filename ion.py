"""Nightingale-specific IO"""

def form_nightingale_property_file(par, isnap):
    """Form the property file name for a given snapshot."""
    catalogue_name = par['Output']['CatalogueName']
    catalogue_name = catalogue_name.replace('XXXX', f'{isnap:04d}')
    return par['Output']['Directory'] + catalogue_name


def form_nightingale_id_file(par, isnap):
    """Form the subhalo ID file name for a given snapshot."""
    id_name = par['Output']['IDFileName']
    id_name = id_name.replace('XXXX', f'{isnap:04d}')
    return par['Output']['Directory'] + id_name

class Output:

    def __init__(self, subhaloes, snapshot):
        pass
