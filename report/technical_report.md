# Technical Report

## Scope
- Q1: Profanity Detection
- Q2: Privacy & Compliance Violation (sensitive info shared without prior identity verification)
- Q3: Call Quality Metrics visualization (overtalk % and silence %)

## Implementations

### Pattern Matching (Regex)
- Profanity:
  - Tolerant regex (handles separators like f***, sh!t) plus aggressive normalization to catch obfuscations.
  - Role-aware aggregation (Agent vs Borrower).
- Privacy & Compliance:
  - Sensitive disclosure keywords restricted to actual disclosures (balance, amount due, account number, credit card number, routing number, last four).
  - Verification detection covers requests (verify/confirm + DOB/address/SSN) and plausible borrower-provided details (DOB formats, street-like, SSN).
  - Violation rule: first agent sensitive disclosure before any verification → violation.

### Machine Learning
- Simple Logistic Regression with constrained vocabularies to stabilize behavior.
- Profanity ML decision = (lexicon/regex hit) OR (classifier positive) to avoid missing explicit profanities while keeping an ML pathway for evolution with more data.
- Sensitive and Verification models use small controlled vocabularies; call-level logic mirrors regex.

## Expected Behavior on Samples
- CALL-1001: Profanity No/No; Privacy No violation (verification before disclosure).
- CALL-1002: Profanity Borrower Yes; Privacy No violation.
- CALL-1003: Profanity Borrower Yes; Privacy Violation (agent disclosed balance before verification).

Across typical files, both approaches now align on profanity most of the time. Minor differences can appear if classifier confidence diverges; lexicon fallback ensures explicit profanities are not missed.

## Visualization (Q3)
- Overtalk %: intersection of Agent and Borrower speaking windows / total duration.
- Silence %: time with no speaker / total duration.
- Simple bar visualization.

## Next Steps
- Grow labeled data to allow ML to operate without lexicon fallback.
- Add evaluation metrics (precision/recall) on a validation set.
- Optionally implement an LLM prompt approach when API access is available.
