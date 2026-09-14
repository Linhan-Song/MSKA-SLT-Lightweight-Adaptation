"""Evaluate a saved checkpoint on the development and test sets.

The underlying runner is shared with training. Pass ``--config`` and
``--resume`` exactly as you would for a training run; this wrapper supplies the
evaluation flag.
"""

import sys

from mska_slt_adaptation.training.run_experiment import main


if __name__ == "__main__":
    if "--eval-only" not in sys.argv:
        sys.argv.append("--eval-only")
    main()
