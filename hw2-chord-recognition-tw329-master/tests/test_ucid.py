import os

def test_ucid():
    with open('ucid', 'r') as f:
        data = str(f.readline())
    assert (data != "UCID_GOES_HERE")