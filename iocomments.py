# Text comments for FOF data
fof_comments = {
	'NumberOfSubhaloes':
            'Number of subhaloes belonging to this FOF.',
	
	'FirstSubhalo': (
		'Index of first subhalo belonging to this '
        'FOF. Note that this will be >= 0 even if there is '
        'not a single subhalo in the FOF group, to keep the '
        'list monotonic. See CentralSubhalo for a safe pointer '
        'to the central subhalo, if it exists.'
        ),
    'CentralSubhalo': (
    	'Index of first subhalo belonging to this '
        'FOF. -1 if the FOF has not a single subhalo.'
    )
}


# Text comments for IDs data
ids_comments = {
	'IDs': (
        "IDs of all particles associated with a subhalo. "
        "Particles are sorted by subhalo, then by type, and "
        "then by radial distance from the subhalo centre."
    )
}

subhalo_comments = {}
