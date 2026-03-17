
import sys
import os
from pathlib import Path
import tempfile

from specollate.search import SpeCollateSearch


def main():


    searcher = SpeCollateSearch(device="cpu", precursor_tolerance=7, precursor_tolerance_type="ppm", charge=5)
    searcher.search_with_sample_data( use_preprocessed=True)


if __name__ == '__main__':
    main()
