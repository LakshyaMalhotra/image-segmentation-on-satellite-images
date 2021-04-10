import os
import datetime
import logging
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

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

    def inverse_ohe(self, ohe_labels: np.ndarray) -> np.ndarray:
        """Converts back the one-hot encoded mask to the multiclass segmentation
        mask.

        Args:
        -----
            ohe_labels (np.ndarray): Output or the target array with shape
            `height` x `width` x `n_classes`.

        Returns:
        --------
            np.ndarray: Segmentation mask
        """
        # we don't want our model to predict class 0
        if ohe_labels.shape[-1] != (self.n_classes) - 1:
            raise ValueError(
                f"The last dimension should contain {self.n_classes-1} but got {ohe_labels.shape[-1]}"
            )

        # adding back 1 to include the null class (no-data class)
        inverse_ohe = (np.argmax(ohe_labels, axis=-1) + 1).astype(np.uint8)

        return inverse_ohe

    def class2color(
        self, ohe_labels: np.ndarray, save_file: bool = False
    ) -> None:
        """Create a colored mask from the model output.

        Args:
        -----
            ohe_labels (np.ndarray): Output or the target array with shape
            `height` x `width` x `n_classes`.
            save_file (bool, optional): Whether or not to save the colored image.
            Defaults to False.
        """
        # raise exception if the path to the output directory and file are not provided
        if (self.output_dir is None) or (self.file_prefix is None):
            raise FileNotFoundError(
                "Output directory/file_name is not specified"
            )

        inverse_ohe_img = self.inverse_ohe(ohe_labels)
        colored_mask = np.zeros((inverse_ohe_img.shape), dtype=np.uint8)

        # create a colored mask with the same shape as ground truth with 3 color channels
        colored_mask = np.dstack([colored_mask, colored_mask, colored_mask])

        # fill the colored mask with the corresponding class color
        for i in range(self.n_classes):
            color = self.class_df[self.class_df["class_id"] == i]["color"]
            locs = np.where(inverse_ohe_img == i)
            colored_mask[locs[0], locs[1], :] = tuple(color)

        colored_mask = Image.fromarray(
            colored_mask.astype(np.uint8), mode="RGB"
        )

        # save the colored mask image
        if save_file:
            colored_mask.save(
                os.path.join(
                    self.output_dir, self.file_prefix + "_output_mask.png"
                )
            )


if __name__ == "__main__":
    logger = get_logger()
    logger.info("Start image preprocessing.")
    preprocess = ImageProcessing(n_classes=3)
    print(preprocess.class_df)
    logger.info("One-hot encoding the segmentation masks.")
    test_array = np.random.randint(0, 3, size=(5, 5))
    print(test_array)
    ohe_array = preprocess.make_ohe(test_array)
    logger.info(f"One hot array:\n {ohe_array}")
    logger.info("Inverse one hot encoding")
    inverse_ohe = preprocess.inverse_ohe(ohe_array)
    logger.info(f"Output array:\n {inverse_ohe}")
    logger.info("Generating colored image from the segmentation mask")
    preprocess.class2color(ohe_array)
