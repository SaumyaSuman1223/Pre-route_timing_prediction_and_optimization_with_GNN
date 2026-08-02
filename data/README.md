# data/

Generated artifacts from the Cadence pipeline. Mostly gitignored (regenerate via
cadence/ scripts rather than committing large report files) — a few small SAMPLE
reports are fine to keep for reference/debugging.

- raw_reports/   one preroute + one postroute .rpt per (circuit, config) pair
- graphs/        parsed PyG Data objects (.pt), one per (circuit, config), ready to
                  load directly into src/data/dataset.py
