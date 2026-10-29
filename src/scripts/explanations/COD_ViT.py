#!/usr/bin/env python3
"""COD_ViT: explain saved SAM_c partitions with BPT and Kernel SHAP."""
from experiment_common import create_parser, run


def parse_args():
    return create_parser('COD_ViT').parse_args()


if __name__ == '__main__':
    run('COD_ViT', parse_args())
