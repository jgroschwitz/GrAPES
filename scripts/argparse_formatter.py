import argparse


class SmartFormatter(argparse.HelpFormatter):
    """
    Custom Help Formatter used to split help text when '\n' was
    inserted in it.
    """
    def _split_lines(self, text, width):
        r = []
        for t in text.splitlines(): r.extend(argparse.HelpFormatter._split_lines(self, t, width))
        return r
