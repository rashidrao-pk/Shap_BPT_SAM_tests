#!/usr/bin/env python3
"""Explain frozen BUSI SigLIP predictions using saved SAM partitions."""
from experiment_common import create_parser, run

def parse_args():
    return create_parser('OC_BUSI').parse_args()


if __name__ == '__main__':
    run('OC_BUSI', parse_args())
