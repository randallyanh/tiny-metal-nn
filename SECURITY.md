# Security Policy

## Supported versions

`tmnn` is pre-1.0. Security fixes are applied to `main` and to the most recently tagged release. Older tags are not patched.

| Version | Supported |
|---|---|
| `main` (latest) | ✅ |
| `v0.x` (latest tag) | ✅ |
| Any earlier tag | ❌ |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for a security report.

Email the maintainer at the address listed on the GitHub repository owner page, with the subject line prefix `[tmnn security]`. Include:

- A description of the issue and its impact
- A minimal reproduction (input, expected behavior, actual behavior)
- The commit hash or tag where the issue was observed
- Whether you'd like to be credited in the disclosure

You should receive an acknowledgement within five business days. We aim to either ship a fix or provide a documented mitigation within 30 days for confirmed issues.

If you do not receive a response within 10 business days, please open a non-detailed issue on GitHub asking the maintainers to check their inbox; do not include the vulnerability details in the public issue.

## Disclosure timeline

- Day 0: Report received and acknowledged.
- Day 1–14: Triage, reproduction, severity assessment.
- Day 15–30: Fix developed and tested in a private branch.
- Day 30: Fix published; CVE requested if applicable; reporter credited if they consented.

This timeline assumes a reproducible issue without external coordination. Multi-party issues (e.g., a vulnerability that also affects an upstream dependency such as `nlohmann_json` or `vcpkg`) may take longer.

## Out of scope

The following are not considered security vulnerabilities:

- Performance regressions on benchmark workloads (open a regular issue with reproduction)
- Compile-time errors with non-C++23 compilers (we target C++23 and don't support older standards)
- Bugs in optional plugin modules clearly marked as experimental or research-grade
- Configuration mistakes that require an attacker to already have local code-execution privileges

## Acknowledgements

We credit all confirmed reporters in the release notes for the version that fixes their issue, unless they request otherwise.
