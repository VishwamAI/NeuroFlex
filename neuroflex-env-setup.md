# NeuroFlex Environment Setup Guide

This guide provides step-by-step instructions for setting up the NeuroFlex environment, ensuring compatibility and optimal performance.

## Prerequisites

- Git
- Python 3.8.12
- pyenv (for managing Python versions)
- pip-tools (for managing dependencies)

## Setup Instructions

1. Clone the Repository:
   ```bash
   git clone https://github.com/VishwamAI/NeuroFlex.git
   cd NeuroFlex
   ```

2. Set Up Python Environment:
   - Ensure Python 3.8.12 is installed using `pyenv`:
     ```bash
     pyenv install 3.8.12
     ```
   - Create and activate a virtual environment:
     ```bash
     pyenv virtualenv 3.8.12 neuroflex-env
     pyenv activate neuroflex-env
     ```

3. Install Dependencies:
   - Use `pip-tools` to manage dependencies:
     ```bash
     pip install pip-tools
     pip-sync requirements.txt
     ```

4. Run Tests:
   - Execute the test suite to verify everything is working:
     ```bash
     pytest
     ```

## Troubleshooting

If you encounter any issues during the setup process, please check the following:

- Ensure all prerequisites are correctly installed.
- Verify that you're using the correct Python version (3.8.12).
- Make sure all commands are run from the root directory of the NeuroFlex project.

For further assistance, please open an issue on the GitHub repository.

## Contributing

If you'd like to contribute to NeuroFlex, please read our contributing guidelines in the repository's README.md file.

## Additional Notes

- Consider using a `.env` file to manage environment variables and secrets.
- Regularly update your dependencies to ensure security and compatibility.
- Follow best practices for Python development, including code style and testing.
