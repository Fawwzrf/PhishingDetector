def test_imports_smoke():
    """
    Smoke test sederhana untuk memastikan modul-modul utama bisa di-import.
    """
    from src.config import load_config  # noqa: F401
    from src.data.loader import DataLoader  # noqa: F401
    from src.pipeline.pipeline import ProductionPipeline  # noqa: F401

