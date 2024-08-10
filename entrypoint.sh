#!/bin/bash
set -e

echo "Starting entrypoint script..."

# Check if virtual environment exists
if [ ! -d "/home/appuser/venv" ]; then
    echo "Creating virtual environment..."
    python -m venv /home/appuser/venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source /home/appuser/venv/bin/activate
# Make sure the activation happens for interactive sessions too
echo "source /home/appuser/venv/bin/activate" >> /home/appuser/.bashrc
# Verify activation
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

echo "Entrypoint script completed."

# Keep the container running and execute any command passed to the script
exec "$@"
