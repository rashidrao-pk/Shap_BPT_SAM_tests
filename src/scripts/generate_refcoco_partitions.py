#!/usr/bin/env python3
"""Generate SAM partitions for the images of the configured RefCOCO split (one per image)."""
from generate_coco_partitions import create_parser, run


def parse_args():
    return create_parser(description=__doc__, default_config='RefCOCO_epito').parse_args()


if __name__ == '__main__':
    run(parse_args())
