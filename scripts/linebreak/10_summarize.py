#!/usr/bin/env python3
"""Print the linebreak headline numbers from the pipeline outputs (CPU).

Only numbers whose inputs exist are printed; error values are means over the evaluated seeds.
"""
from __future__ import annotations

from linebreak import figures, paths


def main() -> int:
    numbers = figures.compute_numbers(paths.OUTPUTS_DIR)
    if not numbers:
        print("no outputs found; run the linebreak pipeline first")
        return 1
    for name, value in numbers.items():
        if name.endswith("_sd"):
            continue
        sd = numbers.get(f"{name}_sd")
        text = f"{value}" if isinstance(value, int) else f"{value:.3f}"
        print(f"  {name:30s} {text:>10}" + (f" ± {sd:.2f}" if sd is not None else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
