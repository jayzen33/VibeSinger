import sys
from functools import reduce

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.module import _addindent

from singer.rmvpe.constants import CONST, N_CLASS


def cycle(iterable):
    while True:
        for item in iterable:
            yield item


def summary(model, file=sys.stdout):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            if hasattr(p, "shape"):
                total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        if file is sys.stdout:
            main_str += ", \033[92m{:,}\033[0m params".format(total_params)
        else:
            main_str += ", {:,} params".format(total_params)
        return main_str, total_params

    string, count = repr(model)
    if file is not None:
        if isinstance(file, str):
            file = open(file, "w")
        print(string, file=file)
        file.flush()

    return count


def to_local_average_cents(salience, center=None, thred=0.05):
    """
    find the weighted average cents near the argmax bin
    """

    if not hasattr(to_local_average_cents, "cents_mapping"):
        # the bin number-to-cents mapping
        to_local_average_cents.cents_mapping = (20 * torch.arange(N_CLASS) + CONST).to(salience.device)  # noqa: F405

    if salience.ndim == 1:
        if center is None:
            center = int(torch.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience = salience[start:end]
        product_sum = torch.sum(salience * to_local_average_cents.cents_mapping[start:end])
        weight_sum = torch.sum(salience)
        return product_sum / weight_sum if torch.max(salience) > thred else 0
    if salience.ndim == 2:
        return torch.Tensor([to_local_average_cents(salience[i, :], None, thred) for i in range(salience.shape[0])]).to(
            salience.device
        )

    raise Exception("label should be either 1d or 2d ndarray")


def to_viterbi_cents(salience, thred=0.05):
    # Create viterbi transition matrix
    if not hasattr(to_viterbi_cents, "transition"):
        xx, yy = torch.meshgrid(range(N_CLASS), range(N_CLASS))  # noqa: F405
        transition = torch.maximum(30 - abs(xx - yy), 0)
        transition = transition / transition.sum(axis=1, keepdims=True)
        to_viterbi_cents.transition = transition

    # Convert to probability
    prob = salience.T
    prob = prob / prob.sum(axis=0)

    # Perform viterbi decoding
    path = librosa.sequence.viterbi(prob.detach().cpu().numpy(), to_viterbi_cents.transition).astype(np.int64)

    return torch.Tensor([to_local_average_cents(salience[i, :], path[i], thred) for i in range(len(path))]).to(
        salience.device
    )


def to_local_average_cents_optimized(salience, center=None, thred=0.05):
    """
    find the weighted average cents near the argmax bin (optimized for 2D tensors)
    """
    if not hasattr(to_local_average_cents_optimized, "cents_mapping"):
        # the bin number-to-cents mapping
        to_local_average_cents_optimized.cents_mapping = (20 * torch.arange(N_CLASS) + CONST).to(salience.device)

    if salience.ndim == 1:
        # 一维情况基本不变，因为其操作已经是向量化的
        if center is None:
            center = int(torch.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience_window = salience[start:end]
        if torch.max(salience_window) < thred:
            return 0.0
        product_sum = torch.sum(salience_window * to_local_average_cents_optimized.cents_mapping[start:end])
        weight_sum = torch.sum(salience_window)
        return product_sum / weight_sum

    if salience.ndim == 2:
        # 向量化处理二维情况
        centers = torch.argmax(salience, dim=1)

        # 创建一个滑动的窗口视图，避免循环
        # 使用 F.pad 来处理边界情况
        padded_salience = F.pad(salience, (4, 4), "constant", 0)
        padded_cents_mapping = F.pad(to_local_average_cents_optimized.cents_mapping, (4, 4), "constant", 0)

        # 使用 gather 和 unsqueeze 来高效地提取每个窗口的数据
        indices = torch.arange(-4, 5, device=salience.device).unsqueeze(0) + centers.unsqueeze(1)

        # 提取每个窗口的 salience 和 cents mapping
        # `indices + 4` 是因为我们在前面填充了4个元素
        salience_windows = torch.gather(padded_salience, 1, indices + 4)
        cents_windows = torch.gather(padded_cents_mapping.expand(salience.shape[0], -1), 1, indices + 4)

        # 计算加权和
        product_sum = torch.sum(salience_windows * cents_windows, dim=1)
        weight_sum = torch.sum(salience_windows, dim=1)

        # 应用阈值
        max_salience_in_window, _ = torch.max(salience_windows, dim=1)
        # 创建一个掩码，用于将低于阈值的结果置为0
        mask = max_salience_in_window >= thred

        # 避免除以零
        # 创建一个非常小的正数 epsilon，用于防止除以零
        epsilon = 1e-8
        result = torch.zeros_like(product_sum)
        result[mask] = product_sum[mask] / (weight_sum[mask] + epsilon)

        return result

    raise Exception("salience should be either 1d or 2d tensor")
