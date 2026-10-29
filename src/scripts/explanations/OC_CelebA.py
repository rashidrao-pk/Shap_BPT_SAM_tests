#!/usr/bin/env python3
"""Explain the notebook's CelebModel attribute logits on CelebAMask-HQ."""
from experiment_common import create_parser, run


def parse_args():
    return create_parser('OC_CelebA').parse_args()


if __name__ == '__main__':
    run('OC_CelebA', parse_args())
