# Security policy

## Supported version

Security fixes are applied to the latest version on the `main` branch.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Use GitHub's
private vulnerability reporting feature when it is enabled for this repository.

## Trust boundaries

- The default MLflow server binds to `127.0.0.1`. Do not expose it publicly
  without authentication, TLS and access controls.
- Pickle, cloudpickle and joblib model files can execute code while loading.
  Load only model artifacts produced by a trusted training run.
- The IBM dataset download is accepted only when its SHA-256 checksum and row
  count match the expected values.
- Secrets belong in environment variables or a secret manager, never in the
  repository. `.env`, data, artifacts and MLflow databases are gitignored.
- The Docker image runs training as an unprivileged user.

## Automated controls

GitHub Actions runs `pip-audit`, `pip check`, Ruff and pytest on every push and
pull request. Dependabot checks Python, Docker and GitHub Actions dependencies
weekly. Third-party GitHub Actions are pinned to full commit SHAs.
