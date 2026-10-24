"""Cost estimate before, and a hard spending cap during, every paid command."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field


class BudgetExceeded(Exception):
    pass


@dataclass
class Estimate:
    calls: int = 0
    input_tokens: int = 0
    expected_output_tokens: int = 0
    max_output_tokens: int = 0
    expected_usd: float = 0.0
    max_usd: float = 0.0
    unpriced_models: set[str] = field(default_factory=set)
    lines: list[str] = field(default_factory=list)

    def add(self, other: 'Estimate') -> None:
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.expected_output_tokens += other.expected_output_tokens
        self.max_output_tokens += other.max_output_tokens
        self.expected_usd += other.expected_usd
        self.max_usd += other.max_usd
        self.unpriced_models |= other.unpriced_models
        self.lines += other.lines

    def describe(self) -> str:
        text = '\n'.join(self.lines)
        text += (f'\nTOTAL: {self.calls} API calls, ~{self.input_tokens:,} input tokens, '
                 f'~{self.expected_output_tokens:,} output tokens expected (cap {self.max_output_tokens:,}).'
                 f'\nEstimated cost: ~${self.expected_usd:.3f} expected, ${self.max_usd:.3f} worst case '
                 f'(every reply hits its token cap, no prompt caching).')
        if self.unpriced_models:
            text += (f'\nWARNING: no price for {", ".join(sorted(self.unpriced_models))}; '
                     'priced as the most expensive model in pricing.json.')
        return text


class Budget:
    """Thread-safe running total. `charge` raises BudgetExceeded once the cap is reached,
    and `exhausted` tells workers to stop before starting another call."""

    def __init__(self, max_usd: float) -> None:
        self.max_usd = max_usd
        self.spent = 0.0
        self._lock = threading.Lock()

    @property
    def exhausted(self) -> bool:
        with self._lock:
            return self.spent >= self.max_usd

    def charge(self, usd: float) -> None:
        with self._lock:
            self.spent += usd or 0.0


def check_and_confirm(estimate: Estimate, max_usd: float, yes: bool) -> bool:
    """Prints the estimate. Returns True if the command may spend money."""
    print(estimate.describe())
    print(f'Spending cap (--max-usd): ${max_usd:.2f}')
    if estimate.expected_usd > max_usd:
        print(f'REFUSED: the expected cost ${estimate.expected_usd:.3f} is above --max-usd ${max_usd:.2f}. '
              'Raise --max-usd or shrink the run (--limit, fewer candidates).')
        return False
    if not yes:
        print('Dry run: nothing was sent. Re-run with --yes to spend up to the cap.')
        return False
    if estimate.max_usd > max_usd:
        print('Note: the worst case is above the cap; the run stops early if actual spend reaches it.')
    return True
