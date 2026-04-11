# Failure Analysis Template

## Setup

- Dataset:
- Split:
- Methods:
- Random seeds:

## Main empirical pattern

- Which method wins on mean final best-so-far?
- Which method selects higher-sigma points?
- Does higher sigma convert into higher realized fitness?

## Shift diagnostics

- Train-candidate embedding distance:
- Support overlap proxy:
- Selected-to-train distance over time:
- Selected-to-candidate shift gap:

## Mechanistic interpretation

1. Closed-loop querying gradually moves the decision process away from the original training support.
2. In high-shift regions, uncalibrated sigma no longer tracks true prediction error reliably.
3. UCB can therefore spend budget on epistemically attractive but low-value regions.

## Cross-split comparison

- Does failure get stronger under stronger shift?
- Does low-resource behave differently from mutation extrapolation?
- Are the conclusions stable across random seeds?

## Paper-facing takeaway

- State clearly whether the evidence supports "uncertainty failure under feedback shift."
