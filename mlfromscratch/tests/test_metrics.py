from mlfromscratch.utils.metrics import root_mean_square_error, mean_absolute_error

y1 = [3, -0.5, 2, 7]
y2 = [2.5, 0.0, 2, 8]


def test_metrics():
    assert root_mean_square_error(y1, y2) == 0.6123724356957945
    assert mean_absolute_error(y1, y2) == 0.5
