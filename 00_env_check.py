#!/usr/bin/env python3
"""
Environment check script for Reddit data analysis pipeline.
Checks Python version and required packages.
"""

import argparse
import importlib
import subprocess
import sys
from pathlib import Path


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("WARNING: Python 3.8+ recommended")
    else:
        print("✓ Python version OK")

def check_package(package_name, min_version=None):
    """Check if package is installed and optionally version."""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package_name}: {version}")
        return True
    except ImportError:
        print(f"✗ {package_name}: NOT INSTALLED")
        return False

def install_package(package_name):
    """Try to install package."""
    try:
        print(f"Attempting to install {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✓ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Failed to install {package_name}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Check environment for Reddit analysis")
    parser.add_argument("--config", type=str, help="Config file path")
    args = parser.parse_args()

    print("=== Environment Check ===")

    # Check Python version
    check_python_version()

    # Check required packages
    required_packages = [
        "pandas",
        "numpy",
        "pyarrow",
        "fastparquet",
        "tqdm",
        "sklearn",
        "matplotlib",
        "networkx",
        "joblib",
        "pydantic",
        "ruamel.yaml"
    ]

    missing_packages = []

    for package in required_packages:
        if not check_package(package):
            missing_packages.append(package)

    # Try to install missing packages
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Attempting to install missing packages...")

        for package in missing_packages:
            if package == "sklearn":
                install_package("scikit-learn")
            else:
                install_package(package)

    # Final check
    print("\n=== Final Check ===")
    all_ok = True
    for package in required_packages:
        if not check_package(package):
            all_ok = False

    if all_ok:
        print("\n✓ All required packages are available")
        return 0
    else:
        print("\n✗ Some packages are still missing")
        return 1

if __name__ == "__main__":
    sys.exit(main())
