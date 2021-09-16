import os

import numpy as np
import tvm
import tvm.relay.testing
from tvm import relay
from tvm.contrib import graph_executor
from tvm.ir import IRModule
from tvm.relay import analysis
from tvm.relay import expr as _expr
from tvm.relay import function as _function
from tvm.relay import op as _op
from tvm.relay import vision as _vision
from tvm.relay.frontend.common import infer_shape as _infer_shape


def mtl_build(wrong_output=False):
    os.environ["PATH"] = os.environ["PATH"] + ":/usr/local/cuda/bin/"
    data = np.load('mtl_dump.npz')

    input_dict = {}
    if wrong_output:
        fpn2_add = _expr.var('fpn2/add', shape=data['fpn2/add'].shape)
        fpn2_dw = _op.nn.conv2d(
            data=fpn2_add,
            weight=_expr.const(data['fpn2/dw_f']),
            strides=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=32,
            kernel_size=3,
            channels=32,
        )
        fpn2_dw = _op.nn.bias_add(fpn2_dw, _expr.const(data['fpn2/dw_b']))
        fpn2_dw = _op.nn.relu(fpn2_dw)
        input_dict['fpn2/add'] = data['fpn2/add']
    else:
        fpn2_dw = _expr.var('fpn2/dw', shape=data['fpn2/dw'].shape)
        input_dict['fpn2/dw'] = data['fpn2/dw']

    fpn2_sep = _op.nn.conv2d(
        data=fpn2_dw,
        weight=_expr.const(data['fpn2/sep_f']),
        strides=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        kernel_size=1,
        channels=32,
    )
    fpn2_sep = _op.nn.bias_add(fpn2_sep, _expr.const(data['fpn2/sep_b']))

    fpn2_box_loc = _op.nn.conv2d(
        data=fpn2_sep,
        weight=_expr.const(data['fpn2_box/loc_f']),
        strides=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        groups=1,
        kernel_size=3,
        channels=8,
    )
    fpn2_box_loc = _op.nn.bias_add(fpn2_box_loc, _expr.const(data['fpn2_box/loc_b']))

    fpn2_box_loc_perm = _op.transpose(fpn2_box_loc, [0, 2, 3, 1])
    fpn2_box_loc_flat = _op.nn.batch_flatten(fpn2_box_loc_perm)

    flat3 = _expr.const(data['fpn3_box/loc/flat'])
    flat4 = _expr.const(data['fpn4_box/loc/flat'])
    flat5 = _expr.const(data['conv5_box/loc/flat'])

    loc_pred = _op.concatenate([fpn2_box_loc_flat, flat3, flat4, flat5], 1)
    cls_prob = _expr.const(data['mbox_conf_softmax'])
    cls_prob = _op.transpose(cls_prob, [0, 2, 1])

    mbox_priorbox = data['mbox_priorbox']
    anchors = _expr.const(mbox_priorbox[0, :1].reshape(1, -1, 4))

    print(_infer_shape(cls_prob))
    print(_infer_shape(loc_pred))
    print(_infer_shape(anchors))
    decoded = _vision.multibox_transform_loc(
        cls_prob=cls_prob,
        loc_pred=loc_pred,
        anchor=anchors,
        clip=True,
        threshold=0.02,
        variances=(0.1, 0.1, 0.2, 0.2),
    )

    nms = _vision.non_max_suppression(
        data=decoded[0],
        valid_count=decoded[1],
        indices=decoded[1],
        max_output_size=5,
        iou_threshold=0.4,
        top_k=20,
        return_indices=False
    )

    indices = _expr.const(np.fromiter(range(5), np.int32), 'int32')
    nms = _op.take(nms, indices, axis=1, mode='fast')

    func = _function.Function(analysis.free_vars(nms), nms)
    mod = IRModule.from_expr(func)

    target = "cuda"
    with tvm.transform.PassContext(opt_level=5):
        lib = relay.build(mod, target)

    # print(lib.imported_modules[0].get_source())

    ctx = tvm.cuda(0)
    m = graph_executor.GraphModule(lib["default"](ctx))

    m.run(**input_dict)
    print(m.get_output(0).asnumpy())


if __name__ == '__main__':
    mtl_build(True)
