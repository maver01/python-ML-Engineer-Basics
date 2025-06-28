import os
import pytest
from testdrivendevelopment.tdd import inc

@pytest.fixture
def value_fixture():
    return 4

def test_answer1(value_fixture):
    assert inc(3) == 4 # without fixture

@pytest.mark.skipif( # if running ```SKIP2=1 poetry run pytest``` it will skip this test
    os.environ.get("SKIP2") == "1",
    reason="SKIP if unnecessary"
)
def test_answer2(value_fixture):
    assert inc(3) == value_fixture # with fixture

@pytest.mark.smoke # defined in the pytest.ini
def test_answer3():
    assert inc(3) == 4 # without fixture