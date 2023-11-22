import numpy as np


def get_box_plot_coords(n_classes, n_hue):
    # Per class there is a space of 0.8 without space between each hue.
    # https://stackoverflow.com/questions/44876759/obtaining-the-exact-data-coordinates-of-seaborn-boxplot-boxes
    # Question by: https://stackoverflow.com/users/4942812/mpa
    # Answered by:
    #   https://stackoverflow.com/users/4124317/importanceofbeingernest
    left_borders = (
        np.array(
            [
                j + (0.1 + i * (0.8 / n_hue))
                for j in range(n_classes)
                for i in range(n_hue)
            ]
        )
        / n_classes
    )
    right_borders = (
        np.array(
            [
                j + (0.1 + (i + 1) * (0.8 / n_hue))
                for j in range(n_classes)
                for i in range(n_hue)
            ]
        )
        / n_classes
    )
    return left_borders, right_borders
