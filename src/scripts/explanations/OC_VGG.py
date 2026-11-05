#!/usr/bin/env python3
"""Explain an ImageNet VGG16 classifier using BPT, SAM_c, and optional baselines."""
from experiment_common import create_parser, run


def parse_args():
    return create_parser('OC_VGG').parse_args()


if __name__ == '__main__':
    run('OC_VGG', parse_args())
