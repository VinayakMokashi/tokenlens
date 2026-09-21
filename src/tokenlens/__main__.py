"""Allow ``python -m tokenlens``."""

import sys

from .cli import main

sys.exit(main())
