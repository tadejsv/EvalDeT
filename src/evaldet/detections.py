class Detections:
    pass

    @classmethod
    def from_coco(cls) -> "Detections":
        pass

    @classmethod
    def from_yolo(cls) -> "Detections":
        pass

    @classmethod
    def from_pascal_voc(cls) -> "Detections":
        pass

    @classmethod
    def from_parquet(cls) -> "Detections":
        pass

    def to_coco(self) -> None:
        pass

    def to_pascal_voc(self) -> None:
        pass

    def to_yolo(self) -> None:
        pass

    def to_parquet(self) -> None:
        pass
