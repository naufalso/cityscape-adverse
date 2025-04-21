# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class CityMappilaryV2(CityscapesDataset):
    """CityMappilaryV2 dataset."""

    def __init__(self,
                 img_suffix='_original.png',
                 seg_map_suffix='_mask.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
