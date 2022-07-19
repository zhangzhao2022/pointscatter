import os.path as osp
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class DeepGlobeDataset(CustomDataset):

    CLASSES = ('background', 'vessel')

    PALETTE = [[120, 120, 120], [6, 230, 230]]

    def __init__(self,
                 **kwargs):
        super(DeepGlobeDataset, self).__init__(
            img_suffix='_sat.jpg',
            seg_map_suffix='_mask.png',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)

