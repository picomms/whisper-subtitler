#!/usr/bin/env python
"""
Simple runner script for whisper-subtitler.
"""

import sys

from whisper_subtitler.modules.cli import main

if __name__ == "__main__":
    sys.exit(main())
