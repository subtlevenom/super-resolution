import argparse
from src.core import logger
from src import cli


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        'Color Transfer Translation', 
        formatter_class=cli.RichHelpFormatter
    )
    subparser = parser.add_subparsers(title='Tools', required=True)

    cli.register_parsers(subparser)

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    args.func(args)


if __name__ == '__main__':
    main()
