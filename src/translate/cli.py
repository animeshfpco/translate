import logging

from translate.config import Config
from translate.pipeline import Pipeline


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = Config()
    Pipeline(cfg).run()


if __name__ == "__main__":
    main()
