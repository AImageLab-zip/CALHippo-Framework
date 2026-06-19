"""Convert format utilities for QuPath compatibility."""

import numpy as np


def to_qupath(file_path, nuc_pos_list, nuc_type_list, type_info_dict):
    """Export to QuPath v0.2.3 compatible format.

    Args:
        file_path: Output file path (.tsv)
        nuc_pos_list: List of nuclei positions
        nuc_type_list: List of nuclei types
        type_info_dict: Dictionary mapping type_id to (name, color)
    """

    def rgb2int(rgb):
        r, g, b = rgb
        return (r << 16) + (g << 8) + b

    nuc_pos_list = np.array(nuc_pos_list)
    nuc_type_list = np.array(nuc_type_list)
    assert nuc_pos_list.shape[0] == nuc_type_list.shape[0]

    with open(file_path, "w") as fptr:
        fptr.write("x\ty\tclass\tname\tcolor\n")

        nr_nuc = nuc_pos_list.shape[0]
        for idx in range(nr_nuc):
            nuc_type = nuc_type_list[idx]
            nuc_pos = nuc_pos_list[idx]
            type_name = type_info_dict[nuc_type][0]
            type_color = type_info_dict[nuc_type][1]
            type_color = rgb2int(type_color)
            fptr.write(
                "{x}\t{y}\t{type_class}\t{type_name}\t{type_color}\n".format(
                    x=nuc_pos[0],
                    y=nuc_pos[1],
                    type_class="",
                    type_name=type_name,
                    type_color=type_color,
                )
            )
    return
