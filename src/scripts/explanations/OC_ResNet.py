#!/usr/bin/env python3
"""OC_ResNet: explain saved SAM_c partitions with BPT and Kernel SHAP."""
from experiment_common import create_parser, run


def parse_args():
    return create_parser('OC_ResNet').parse_args()


if __name__ == '__main__':
    run('OC_ResNet', parse_args())
