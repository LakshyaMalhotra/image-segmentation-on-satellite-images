import os
import datetime
import logging
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

# settings
warnings.filterwarnings("ignore")

# logger
def get_logger():
    FORMAT = "[%(levelname)s]%(asctime)s:%(name)s:%(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger("preprocessing")
    logger.setLevel(logging.DEBUG)
    return logger


class ImageProcessing:
    """Basic image processing on input and output images."""

    def __init__(
        self,
        n_classes: int = 11,
        file_prefix: str = None,
        output_dir: str = None,
    ):
        """Class constructor creating a dataframe for input labels describing
        different land cover classes.

        Args:
        -----
            n_classes (int, optional): Number of different classes. Defaults to 11.
            file_prefix (str, optional): File prefix if any. Defaults to None.
            output_dir (str, optional): Path to the output directory. Defaults to None.
        """
        # List of tuples showing the color with the class id
        self.labels = [
            ("label", "class_id", "color"),
            ("No data", 0, (0, 0, 0)),
            ("Trees cover areas", 1, (0, 160, 0)),
            ("Shrubs cover areas", 2, (150, 100, 0)),
            ("Grassland", 3, (255, 180, 0)),
            ("Cropland ", 4, (255, 255, 100)),
            ("Vegetation Aquatic or regularly flooded", 5, (0, 220, 130)),
            ("Lichen Mosses/Sparse vegetation", 6, (255, 235, 175)),
            ("Bare areas", 7, (255, 245, 215)),
            ("Built up areas", 8, (195, 20, 0)),
            ("Snow or ice or clouds", 9, (255, 255, 255)),
            ("Open water", 10, (0, 70, 200)),
        ]
        self.class_df = pd.DataFrame(self.labels[1:], columns=self.labels[0])
        self.file_prefix = file_prefix
        self.output_dir = output_dir
        self.n_classes = n_classes

    def make_ohe(self, mask_array: np.ndarray) -> np.ndarray:
        """Create one-hot encoded mask with shape: `height` x `width` x `n_classes`
        from the segmentation mask.

        Args:
        -----
            mask_array (np.ndarray): 2D array containing the segmentation mask
            for each image. Each element corresponds to the class label for that
            particular pixel.

        Returns:
        --------
            np.ndarray: One-hot encoded mask with shape: `height` x `width` x `n_classes`
        """
        height, width = mask_array.shape
        one_hot = np.zeros((height, width, self.n_classes), dtype=int)

        # iterate over all the classes
        for i in range(self.n_classes):
            x, y = np.where(mask_array == i)
            one_hot[x, y, i] = 1

        return one_hot


if __name__ == "__main__":
    logger = get_logger()
    logger.info("Start image preprocessing.")
    preprocess = ImageProcessing()
    print(preprocess.class_df)