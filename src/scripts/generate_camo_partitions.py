#!/usr/bin/env python3
"""Generate CAMO Train partitions with SAM2 Tiny and a target of 50 raw masks."""
import os
from pathlib import Path

from generate_coco_partitions import create_parser, run


def parse_args():
    parser = create_parser(description=__doc__, default_config='CAMO_epito')
    sam2_dir = Path(os.environ.get('SAM2_DIR', '/beegfs/home/mrashid/repos/third_party/sam2'))
    parser.set_defaults(model='sam2', generate_at_least_n_masks=50,
                        sam2_config='configs/sam2.1/sam2.1_hiera_t.yaml',
                        checkpoint=sam2_dir / 'checkpoints/sam2.1_hiera_tiny.pt')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
