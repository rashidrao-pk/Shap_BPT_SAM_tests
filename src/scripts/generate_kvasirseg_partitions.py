#!/usr/bin/env python3
"""Generate SAM partitions for the configured KvasirSEG split."""
from generate_coco_partitions import create_parser, run


def parse_args():
    return create_parser(description=__doc__, default_config='KvasirSEG_epito').parse_args()


if __name__ == '__main__':
    run(parse_args())
