import math
import torch.nn.functional as F
import torch

def extract_image_patches(x, kernel, stride=1, dilation=1):
    # print(x.shape)
    batch_size,c,h,w = x.shape
    patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
    # print(patches.shape)
    patches = patches.permute(0,4,5,1,2,3).contiguous()
    return patches.view(batch_size, -1, patches.shape[-2], patches.shape[-1])

def extract_patches(x,
                    sizes,
                    strides=[1, 1, 1, 1],
                    rates=[1, 1, 1, 1],
                    padding='VALID'):
    b,h,w,c = x.shape
    filter_height = sizes[1]
    filter_width = sizes[2]
    patches = x.unfold(1, filter_height, strides[1]).unfold(2, filter_width, strides[2])
    patches = patches.permute(0, 1, 2, 4, 5, 3).contiguous()
    x_patches =  patches.view(b, patches.shape[1], patches.shape[2], -1)
    return x_patches

def smooth_max(input_tensor, dim = 1, alpha = 1):
    ax = torch.exp(torch.mul(input_tensor, alpha))
    s_max = torch.mul(input_tensor, ax / torch.sum(ax, dim=dim, keepdims=True))
    return torch.sum(s_max, dim=dim)



###########################################################################
import collections
from itertools import repeat
from typing import List, Dict, Any


def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


_single = _ntuple(1, "_single")
_pair = _ntuple(2, "_pair")
_triple = _ntuple(3, "_triple")
_quadruple = _ntuple(4, "_quadruple")


def _reverse_repeat_tuple(t, n):
    r"""Reverse the order of `t` and repeat each element for `n` times.

    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple(x for x in reversed(t) for _ in range(n))


def _list_with_default(out_size: List[int], defaults: List[int]) -> List[int]:
    if isinstance(out_size, int):
        return out_size
    if len(defaults) <= len(out_size):
        raise ValueError(
            "Input dimension should be at least {}".format(len(out_size) + 1)
        )
    return [
        v if v is not None else d for v, d in zip(out_size, defaults[-len(out_size) :])
    ]


def consume_prefix_in_state_dict_if_present(
    state_dict: Dict[str, Any], prefix: str
) -> None:
    r"""Strip the prefix in state_dict in place, if any.

    ..note::
        Given a `state_dict` from a DP/DDP model, a local model can load it by applying
        `consume_prefix_in_state_dict_if_present(state_dict, "module.")` before calling
        :meth:`torch.nn.Module.load_state_dict`.

    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix) :]
            state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata if any.
    if "_metadata" in state_dict:
        metadata = state_dict["_metadata"]
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix) :]
            metadata[newkey] = metadata.pop(key)
