import numpy as np

from visualization import get_box_plot_coords


def test_get_box_plot_coords():
    n_classes = 3
    n_hue = 2
    left_borders, right_borders = get_box_plot_coords(n_classes, n_hue)

    # Check that the returned values are numpy arrays
    assert isinstance(left_borders, np.ndarray)
    assert isinstance(right_borders, np.ndarray)

    # Check that the arrays have the correct length
    assert len(left_borders) == n_classes * n_hue
    assert len(right_borders) == n_classes * n_hue

    # Check that the values in the arrays are as expected
    expected_left_borders = np.array(
        [0.1 / 3, 0.5 / 3, 1.1 / 3, 1.5 / 3, 2.1 / 3, 2.5 / 3]
    )
    expected_right_borders = np.array(
        [0.5 / 3, 0.9 / 3, 1.5 / 3, 1.9 / 3, 2.5 / 3, 2.9 / 3]
    )
    assert np.allclose(left_borders, expected_left_borders)
    assert np.allclose(right_borders, expected_right_borders)
