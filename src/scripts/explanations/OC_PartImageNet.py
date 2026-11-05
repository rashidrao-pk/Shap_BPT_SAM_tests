#!/usr/bin/env python3
"""Explain a frozen ImageNet ResNet50 on PartImageNet images and part masks."""
from experiment_common import create_parser, run


def parse_args():
    return create_parser('OC_PartImageNet').parse_args()


if __name__ == '__main__':
    run('OC_PartImageNet', parse_args())
