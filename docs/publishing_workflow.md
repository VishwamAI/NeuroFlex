# NeuroFlex Package Publishing Workflow

This document provides a detailed guide on how to publish the NeuroFlex package to PyPI. It includes prerequisites, step-by-step instructions, and troubleshooting tips for common issues.

## Prerequisites

Before publishing the package, ensure you have the following:

1. **PyPI Account**: You need a PyPI account to upload the package. If you don't have one, you can create it [here](https://pypi.org/account/register/).
2. **PyPI API Token**: Generate an API token from your PyPI account. This token will be used for authentication during the publishing process. You can generate a token [here](https://pypi.org/manage/account/#api-tokens).
3. **GitHub Secrets**: Add the PyPI API token to your GitHub repository secrets. This will allow the GitHub Actions workflow to access the token securely. Go to your repository settings, navigate to "Secrets and variables" > "Actions", and add a new secret with the name `PYPI_API_TOKEN`.

## Steps for Publishing

Follow these steps to publish the NeuroFlex package to PyPI:

1. **Update Version**: Ensure the version number in `setup.py` is updated. Increment the version number according to semantic versioning (e.g., from `0.1.3` to `0.1.4`).

2. **Create a Tag**: Create a new tag for the release. This will trigger the GitHub Actions workflow to publish the package.
   ```bash
   git tag v0.1.4
   git push origin v0.1.4
   ```

3. **GitHub Actions Workflow**: The `.github/workflows/ci_cd.yml` file contains the workflow for publishing the package. The `publish` job is triggered on push events to tags. It performs the following steps:
   - Checks out the repository
   - Sets up Python 3.8
   - Installs build tools (`build` and `twine`)
   - Verifies the distribution files
   - Publishes the package to PyPI using `twine`

4. **Verify the Release**: After the workflow completes, verify that the package is published to PyPI. You can check the package page on PyPI [here](https://pypi.org/project/neuroflex/).

## Troubleshooting

Here are some common issues and their solutions:

1. **Authentication Error**: If you encounter an authentication error during the publishing process, ensure that the PyPI API token is correctly added to the GitHub repository secrets and that it has the necessary permissions.

2. **Version Conflict**: If you receive an error about a version conflict, ensure that the version number in `setup.py` is unique and has not been used in a previous release.

3. **Build Errors**: If there are errors during the build process, check the `setup.py` file for any issues with the package metadata or dependencies. Ensure that all required files are included in the package.

4. **Network Issues**: If the publishing process fails due to network issues, try rerunning the GitHub Actions workflow. You can do this from the "Actions" tab in your GitHub repository.

By following these steps and troubleshooting tips, you should be able to successfully publish the NeuroFlex package to PyPI.
