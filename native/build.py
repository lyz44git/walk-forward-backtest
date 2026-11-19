"""Compile native/qmb_kernels.cpp into native/build/libqmb_kernels.so.

Usage: python3 native/build.py            (uses $CXX, default c++)
No -ffast-math; -ffp-contract=off keeps every floating-point operation as written in the source.
"""
from pathlib import Path
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "qmb_kernels.cpp"
LIBRARY = HERE / "build" / "libqmb_kernels.so"
FLAGS = ["-std=c++17", "-O2", "-ffp-contract=off", "-fPIC", "-shared", "-Wall", "-Wextra", "-Wpedantic"]


def compiler() -> str:
    return os.environ.get("CXX", "c++")


def build() -> Path:
    LIBRARY.parent.mkdir(exist_ok=True)
    subprocess.run([compiler(), *FLAGS, str(SOURCE), "-o", str(LIBRARY)], check=True)
    return LIBRARY


if __name__ == "__main__":
    try:
        print(build())
    except (OSError, subprocess.CalledProcessError) as error:
        sys.exit(f"native build failed: {error}")
