import abc

import numpy as np


class MOTMetricBase(abc.ABC):
    @abc.abstractmethod
    def _get_iou_frame(self, frame: int) -> np.ndarray:
        pass
