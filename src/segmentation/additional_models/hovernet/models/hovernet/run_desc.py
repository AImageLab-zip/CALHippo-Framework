"""Inference step for HoverNet."""

from collections import OrderedDict

import torch
import torch.nn.functional as F


def infer_step(batch_data, model):
    """Run inference on a batch of patches.
    
    Args:
        batch_data: Input batch of image patches
        model: HoverNet model
        
    Returns:
        Concatenated prediction output (np, hv, and optionally tp)
    """
    patch_imgs = batch_data

    patch_imgs_gpu = patch_imgs.to("cuda").type(torch.float32)
    patch_imgs_gpu = patch_imgs_gpu.permute(0, 3, 1, 2).contiguous()

    model.eval()

    with torch.no_grad():
        pred_dict = model(patch_imgs_gpu)
        pred_dict = OrderedDict(
            [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
        )
        pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1:]
        if "tp" in pred_dict:
            type_map = F.softmax(pred_dict["tp"], dim=-1)
            type_map = torch.argmax(type_map, dim=-1, keepdim=True)
            type_map = type_map.type(torch.float32)
            pred_dict["tp"] = type_map
        pred_output = torch.cat(list(pred_dict.values()), -1)

    return pred_output.cpu().numpy()
