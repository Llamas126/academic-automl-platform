import logging
import sys

from app.core.config import get_settings


def configure_logging() -> None:
    level = logging.DEBUG if get_settings().debug else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    for noisy in ("matplotlib", "fontTools", "shap", "numba", "PIL", "pylatex"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
