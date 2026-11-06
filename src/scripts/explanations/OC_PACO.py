#!/usr/bin/env python3
"""Explain a frozen ImageNet ResNet50 on PACO-LVIS images and part masks."""
from experiment_common import create_parser, run


def parse_args():
    return create_parser('OC_PACO').parse_args()


if __name__ == '__main__':
    run('OC_PACO', parse_args())
