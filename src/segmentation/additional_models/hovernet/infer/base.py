"""Base inference manager for HoverNet."""

import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..models.hovernet import net_desc, post_proc, run_desc


def convert_pytorch_checkpoint(net_state_dict):
    """Convert data-parallel checkpoint to single GPU mode."""
    variable_name_list = list(net_state_dict.keys())
    is_in_parallel_mode = all(v.split(".")[0] == "module" for v in variable_name_list)
    if is_in_parallel_mode:
        print("WARNING: Detect checkpoint saved in data-parallel mode. Converting...")
        net_state_dict = {
            ".".join(k.split(".")[1:]): v for k, v in net_state_dict.items()
        }
    return net_state_dict


class InferManager(object):
    """Base inference manager class."""

    def __init__(self, **kwargs):
        self.run_step = None
        for variable, value in kwargs.items():
            self.__setattr__(variable, value)
        self.__load_model()
        self.nr_types = self.method["model_args"]["nr_types"]

        # default type info
        self.type_info_dict = {
            None: ["no label", [0, 0, 0]],
        }

        if self.nr_types is not None and self.type_info_path is not None:
            self.type_info_dict = json.load(open(self.type_info_path, "r"))
            self.type_info_dict = {
                int(k): (v[0], tuple(v[1])) for k, v in self.type_info_dict.items()
            }
            for k in range(self.nr_types):
                if k not in self.type_info_dict:
                    assert False, "Not detect type_id=%d defined in json." % k

        if self.nr_types is not None and self.type_info_path is None:
            cmap = plt.get_cmap("hot")
            colour_list = np.arange(self.nr_types, dtype=np.int32)
            colour_list = (cmap(colour_list)[..., :3] * 255).astype(np.uint8)
            self.type_info_dict = {
                k: (str(k), tuple(v)) for k, v in enumerate(colour_list)
            }
        return

    def __load_model(self):
        """Create the model and load checkpoint."""
        model_creator = net_desc.create_model

        net = model_creator(**self.method["model_args"])
        saved_state_dict = torch.load(self.method["model_path"])["desc"]
        saved_state_dict = convert_pytorch_checkpoint(saved_state_dict)

        net.load_state_dict(saved_state_dict, strict=True)
        net = torch.nn.DataParallel(net)
        net = net.to("cuda")

        infer_step = run_desc.infer_step
        self.run_step = lambda input_batch: infer_step(input_batch, net)

        self.post_proc_func = post_proc.process
        return

    def __save_json(self, path, old_dict, mag=None):
        """Save prediction results to JSON."""
        new_dict = {}
        for inst_id, inst_info in old_dict.items():
            new_inst_info = {}
            for info_name, info_value in inst_info.items():
                if isinstance(info_value, np.ndarray):
                    info_value = info_value.tolist()
                new_inst_info[info_name] = info_value
            new_dict[int(inst_id)] = new_inst_info

        json_dict = {"mag": mag, "nuc": new_dict}
        with open(path, "w") as handle:
            json.dump(json_dict, handle)
        return new_dict
