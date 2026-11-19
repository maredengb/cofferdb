---
name: Bug report
about: Report a reproducible CofferDB problem using sanitized data
---

## Environment

Include the CofferDB version or commit, OS and architecture, filesystem,
resource profile, budget overrides, and durability mode. For Python, include
the Python version. If you built from source, include the Rust version.

## Reproduction

Include a minimal CLI command, Rust example, or Python example that reproduces
the problem. Provide a small, sanitized input and say whether the problem occurs
on a newly created database or an existing one.

## Expected and actual behavior

Include the relevant result, exit code and sanitized error/check output. For
ranking problems, distinguish an incorrect calculation from an application
relevance concern and identify the query, weights and supplied vector source.

## Privacy check

Do not attach confidential documents, vectors, credentials, private paths or a
complete private database. Follow SECURITY.md for a security-sensitive report;
ordinary issues are public.
