import re


def sanitize_filename(fname):
    """Sanitize the a filename.

    Allowed characters in a file name are: ``A-Z``, ``a-z``, ``0-9``, ``-``,
    ``_``, and ``.``. All non-allowed characters (including spaces) are
    replaced with an underscore.

    Parameters
    ----------
    fname : str or path-like
        The filename to sanitize.

    Returns
    -------
    sanitized_fname : str
        The sanitized filename.

    Raises
    ------
    ValueError
        If nothing's left after sanitization.
    """
    fname = str(fname)
    sanitized_fname = re.sub(r'[^\w\-_\.]', '', fname)

    if not sanitized_fname:
        msg = ('No filename left after sanitization! Please reconsider your '
               'naming scheme.')
        raise ValueError(msg)

    return sanitized_fname
