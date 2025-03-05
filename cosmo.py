"""Cosmological routines."""

from astropy.cosmology import Planck18
from pdb import set_trace

# Set the cosmology here. By default we use Planck18, but this should make
# it easy to change it for other scenarios.
cosmology = Planck18

def compute_hubble_z(redshift):
    """Compute the Hubble factor for the specified redshift."""
    hubble_z = cosmology.H(redshift)
    if not hubble_z.unit == 'km / (s Mpc)':
        print("Hubble constant seems to be in non-standard units...")
        set_trace()
    print(f"Determined H(z) = {hubble_z.value:.2f} km/s/Mpc.")
    return hubble_z.value


def compute_softening_length(redshift):
    return 0.0007
