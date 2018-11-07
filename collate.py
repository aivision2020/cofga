from torch._six import string_classes, int_classes, FileNotFoundError
import torch
import collections
import re
import numpy as np


def calc_out_dims(batch):
    hs = [x.shape[1] for x in batch]
    ws = [x.shape[2] for x in batch]
    max_h = int(max(hs))
    max_w = int(max(ws))
    return (len(batch), 3, max_h, max_w)


def get_start_end(orig_shape, out_shape):
    _, h, w = orig_shape
    _, _, oh, ow = out_shape
    ohh = np.round(oh / 2.)
    oww = np.round(ow / 2.)
    hh = np.round(h / 2.)
    ww = np.round(w / 2.)
    sh = ohh-hh
    eh = ohh+hh
    sw = oww-ww
    ew = oww+ww
    return int(sh), int(eh), int(sw), int(ew)


def collate_tensors(batch):
    out_dims = calc_out_dims(batch)
    out_variable = batch[0].new(*out_dims).fill_(0)
    for j in range(out_dims[0]):
        sh, eh, sw, ew = get_start_end(batch[j].shape, out_dims)
        out_variable[j, :, sh:eh, sw:ew] = batch[j]
    return out_variable


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        return collate_tensors(batch)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))