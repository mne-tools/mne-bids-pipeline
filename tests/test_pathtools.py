import sys
sys.path.insert(0, '..')

import pytest  # noqa:E402
from pathtools import sanitize_filename


@pytest.mark.parametrize(
    ('input_fname', 'expected_sanitized_fname', 'should_raise'),
    [('all_good_here', 'all_good_here', False),
     ('please/sanitize', 'pleasesanitize', False),
     ('we-are_all.allowed-123ABC', 'we-are_all.allowed-123ABC', False),
     (r':\/,#?=+()*^%$@!<>^', None, True)])
def test_sanitize_filename(input_fname, expected_sanitized_fname,
                           should_raise):
    if should_raise:
        with pytest.raises(ValueError, match='No filename left'):
            sanitized_fname = sanitize_filename(input_fname)
    else:
        sanitized_fname = sanitize_filename(input_fname)
        assert sanitized_fname == expected_sanitized_fname
