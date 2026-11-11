"""Настройка логирования."""

import logging


def setup_logging() -> None:
    # basicConfig ничего не меняет, если логирование уже настроено (например, в тестах).
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


logger = logging.getLogger("churn")
