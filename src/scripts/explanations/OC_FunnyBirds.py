#!/usr/bin/env python3
"""Explain the FunnyBirds authors' frozen ResNet-50 on synthetic birds and part masks."""
from experiment_common import create_parser, run


def parse_args():
    return create_parser('OC_FunnyBirds').parse_args()


if __name__ == '__main__':
    run('OC_FunnyBirds', parse_args())
