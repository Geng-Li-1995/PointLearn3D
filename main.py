"""PointLearn3D entry point."""

from config.input import Input
from config.config import run


def main():
    run(Input())


if __name__ == "__main__":
    main()
