import pytest

OPEN_STORES: list = []  # stores opened by helpers that cannot own a fixture (e.g. tests.test_capture.deps)


@pytest.fixture(autouse=True)
def _close_helper_stores():
    yield
    while OPEN_STORES:
        OPEN_STORES.pop().close()


@pytest.fixture(autouse=True)
def _no_pair_wait(monkeypatch):
    """Comment+forward gluing waits CLAWDIA_PAIR_WINDOW_S for a counterpart; tests that exercise it set the window themselves."""
    monkeypatch.setenv("CLAWDIA_PAIR_WINDOW_S", "0")
