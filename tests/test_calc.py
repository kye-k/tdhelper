import pytest
from tdhelper.core import TrajectoryData
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

@pytest.fixture(scope="session")
def spt_data():
    td = TrajectoryData()
    td.import_trc(EXAMPLES_DIR / "data" / "dyn1bb_ctrl_1_stream1_trcPALMTracer.txt", 0.106)
    return td

def test_import(spt_data):
    assert len(spt_data.data) == 32231

def test_DCOEF(spt_data):
    assert 1 == 1
    #median Dapp: 0.00522044853686294
    #mean Dapp: 0.05049754223697419