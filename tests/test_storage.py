import os
import pytest
from src.metric_calculator import ModelContext
from src.storage_manager import StorageManager

def test_storage_manager_download(monkeypatch, tmp_path):
    """
    Test StorageManager downloads files and updates ModelContext.
    """

    # Fake requests.get to avoid real HTTP calls
    class FakeResponse:
        def __init__(self, content):
            self.content = content
        def raise_for_status(self):
            pass

    def fake_get(url):
        return FakeResponse(b"fake content")

    # Patch requests.get inside storage_manager
    import src.storage_manager as storage_module
    monkeypatch.setattr(storage_module.requests, "get", fake_get)

    context = ModelContext(
        model_url="http://fakeurl/model.pt",
        dataset_url="http://fakeurl/data.csv",
        model_info={"size": 10}
    )
    storage = StorageManager(base_path=tmp_path)
    storage.store_model(context)

    # Assert local paths are set
    assert context.local_model_path.endswith("model.pt")
    assert context.local_dataset_path.endswith("data.csv")
    # Assert files exist
    assert os.path.exists(context.local_model_path)
    assert os.path.exists(context.local_dataset_path)
