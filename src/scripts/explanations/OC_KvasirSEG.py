#!/usr/bin/env python3
"""Explain a frozen Kvasir v2 ViT on Kvasir-SEG polyps."""
from experiment_common import create_parser, run


def parse_args():
    return create_parser('OC_KvasirSEG').parse_args()


if __name__ == '__main__':
    run('OC_KvasirSEG', parse_args())
