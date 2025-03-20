import sys
import os
import importlib.util

# Print Python executable path
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

# Check if crawl4ai is installed
crawl4ai_spec = importlib.util.find_spec("crawl4ai")
if crawl4ai_spec is not None:
    print(f"crawl4ai is installed at: {crawl4ai_spec.origin}")
    import crawl4ai
    print(f"crawl4ai version: {crawl4ai.__version__}")
else:
    print("crawl4ai is NOT installed in this environment")

# Print current environment variables
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
print(f"CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', 'Not set')}")
