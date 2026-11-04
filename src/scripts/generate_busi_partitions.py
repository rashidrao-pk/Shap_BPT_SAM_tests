#!/usr/bin/env python3
"""Generate BUSI SAM1/SAM2 partitions, excluding lesion annotation images."""
from generate_coco_partitions import create_parser, run

def parse_args():
    return create_parser(description=__doc__, default_config='BUSI_epito').parse_args()


if __name__ == '__main__':
    run(parse_args())
