from .btc_center import Btc_Center
from .btc_center_onestage import Btc_Center_Onestage
from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .point_rcnn_symmetry import POINTRCNN_symmetry
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .pv_rcnn_symmetry import PVRCNN_symmetry
from .second_net import SECONDNet
from .second_net_iou import SECONDNetIoU
from .caddn import CaDDN
from .voxel_rcnn import VoxelRCNN
from .centerpoint import CenterPoint
from .pv_rcnn_plusplus import PVRCNNPlusPlus
from .sydetector_point_rcnn import PointRCNN_cls

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'SECONDNetIoU': SECONDNetIoU,
    'CaDDN': CaDDN,
    'VoxelRCNN': VoxelRCNN,
    'CenterPoint': CenterPoint,
    'PVRCNNPlusPlus': PVRCNNPlusPlus,
    'PointRCNN_cls': PointRCNN_cls,
    'PVRCNN_symmetry': PVRCNN_symmetry,
    'POINTRCNN_symmetry': POINTRCNN_symmetry,
    'Btc_Center': Btc_Center,
    'Btc_Center_Onestage': Btc_Center_Onestage,
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
