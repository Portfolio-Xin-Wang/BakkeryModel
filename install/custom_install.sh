# Set poetry up.
ECHO "Custom install script."

poetry install --no-root --no-directory
# Then install PyTorch separately to avoid downloading unnecessary binaries.
pip install --no-cache-dir -r install/requirements_torch.txt
