#!/usr/bin/env python3
"""OD_DETR: explain saved SAM_c partitions with BPT and Kernel SHAP."""
from experiment_common import create_parser, run


def parse_args():
    return create_parser('OD_DETR').parse_args()


if __name__ == '__main__':
    run('OD_DETR', parse_args())
