#!/usr/bin/env python3
"""Generate ImageNet partitions with shared SAM tuning, logging, and resume support."""
from generate_coco_partitions import create_parser, run


def parse_args():
    return create_parser(description=__doc__, default_config='ImageNet_epito').parse_args()


if __name__ == '__main__':
    run(parse_args())
