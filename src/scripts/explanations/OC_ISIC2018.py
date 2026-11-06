#!/usr/bin/env python3
"""Explain a frozen ISIC 2018 Task 3 EfficientNet on Task 1 lesion masks."""
from experiment_common import create_parser, run


def parse_args():
    return create_parser('OC_ISIC2018').parse_args()


if __name__ == '__main__':
    run('OC_ISIC2018', parse_args())
