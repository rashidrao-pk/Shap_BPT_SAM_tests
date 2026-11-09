#!/usr/bin/env python3
"""OD_DOTA: explain YOLO11-OBB detections on DOTA tiles with BPT, SAM_c and Kernel SHAP."""
from experiment_common import create_parser, run


def parse_args():
    return create_parser('OD_DOTA').parse_args()


if __name__ == '__main__':
    run('OD_DOTA', parse_args())
