import re
from pathlib import Path

import shap_bpt

ROOT = Path(__file__).resolve().parents[1]


def test_setup_reads_the_package_version():
    assert re.fullmatch(r'\d+\.\d+(\.\d+)?', shap_bpt.__version__)
    assert "VERSION" in (ROOT / 'setup.py').read_text()


def test_build_info_reports_the_extension():
    info = shap_bpt.build_info()
    assert info['version'] == shap_bpt.__version__
    assert info['extension'] and 'bpt' in Path(info['extension']).name
    assert isinstance(info['extension_older_than_source'], bool)
