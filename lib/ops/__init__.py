# This file is added for back-compatibility. Thus, downstream codebase
# could still use and import mmdet.ops.

'''
nms - non maximum suppression: when many boxes overlap on same object, keeps highest
scoring one and drops near duplicates

soft_nms - gentler NMS that lowers overlapping boxes' scores instead of deleting
them; only runs if TEST.SOFT_NMS.ENABLED is on and off by default

roi pooling: cut out each box and resize it to same small size - model looks at
2k boxes of diff sizes but part that decides "is this a dog" only accepts one
fixed size. roi pooling crops each box and squishes it to 7x7 so they all fit;

roi align is just more precise version of roi pooling
'''

# yapf: disable
from torchvision.ops import RoIPool, RoIAlign, nms

def soft_nms(*args):
    # torchvision doesn't have this and isn't used in main path anyways so fine
    raise NotImplementedError("soft_nms not ported")

# yapf: enable
__all__ = [
    'RoIPool', 'RoIAlign', 'nms', 'soft_nms'
]
