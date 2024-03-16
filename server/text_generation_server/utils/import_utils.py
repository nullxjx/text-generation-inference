import os
from loguru import logger
import torch
import torch_npu
import importlib.util

IS_ROCM_SYSTEM = torch.version.hip is not None
IS_CUDA_SYSTEM = torch.version.cuda is not None


def init_torch_npu():
    torch.npu.set_compile_mode(jit_compile=False)
    option = {}
    option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
    torch.npu.set_option(option)

    device_id = int(os.getenv("NPU_VISIBLE_DEVICES", "0"))
    logger.info(f"NPU_VISIBLE_DEVICES:{device_id}")
    torch.npu.set_device(torch.device(f"npu:{device_id}"))

    device_id = torch_npu.npu.current_device()
    logger.info(f"npu is available, device id: {device_id}")


def is_torch_npu_available(check_device=False):
    "Checks if `torch_npu` is installed and potentially if a NPU is in the environment"
    if importlib.util.find_spec("torch_npu") is None:
        return False

    if check_device:
        try:
            # Will raise a RuntimeError if no NPU is found
            _ = torch.npu.device_count()
            result = torch.npu.is_available()
            if result:
                init_torch_npu()
            return result
        except RuntimeError:
            return False
    result = hasattr(torch, "npu") and torch.npu.is_available()
    if result:
        init_torch_npu()
    return result
