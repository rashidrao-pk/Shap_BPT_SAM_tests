#!/usr/bin/env python3
"""Download the E5 notebook's CelebModel weights, verifying SHA-256 before saving."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

URL = ('https://raw.githubusercontent.com/rashidrao-pk/shap_bpt_tests/main/'
       'notebooks/E5_CelebA/models/model.pth')
SHA256 = '0c25db7f75ee66dd7b39bbf99dcddd9a70343deed622337704b270b6f1354892'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def download(output, overwrite=False):
    output = Path(output).expanduser().resolve()
    if output.exists() and not overwrite:
        if digest(output) != SHA256:
            raise ValueError(f'Existing checkpoint checksum differs: {output}; use another path or --overwrite')
        print(f'Checkpoint verified: {output}')
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.part')
    try:
        urllib.request.urlretrieve(URL, temporary)
        if digest(temporary) != SHA256:
            raise ValueError('Downloaded checkpoint checksum differs; destination was not replaced')
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Checkpoint downloaded and verified: {output}')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'checkpoints/celeba_model.pth')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    download(args.output, args.overwrite)
