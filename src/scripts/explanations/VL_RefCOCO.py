#!/usr/bin/env python3
"""VL_RefCOCO: explain CLIP's image-text similarity for RefCOCO referring expressions with BPT, SAM_c and Kernel SHAP."""
from experiment_common import create_parser, run


def parse_args():
    return create_parser('VL_RefCOCO').parse_args()


if __name__ == '__main__':
    run('VL_RefCOCO', parse_args())
