import pytest
import subsync


@pytest.mark.parametrize('s1, s2, true_offset', [
    ('111001', '11001', -1),
    ('1001', '1001', 0),
    ('10010', '01001', 1)
])
def test_fft_alignment(s1, s2, true_offset):
    assert subsync.get_best_offset(s1, s2) == true_offset
