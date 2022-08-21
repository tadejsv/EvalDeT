import abc

import numpy as np


class MOTMetricBase(abc.ABC):
    """A base class for MOT metric classes.

    It declares a method to get the IoU scores for a frame - this is to enable efficient
    sharing of the IoU scores between the different metrics.
    """

    @abc.abstractmethod
    def _get_iou_frame(self, frame: int) -> np.ndarray:
        pass
