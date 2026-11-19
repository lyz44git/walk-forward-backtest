"""Compile native/qmb_kernels.cpp into native/build/libqmb_kernels.so and native/itch_book.cpp into
native/build/itch_book (the NASDAQ ITCH 5.0 order-book parser of Part 6).

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
ITCH_SOURCE = HERE / "itch_book.cpp"
ITCH_BINARY = HERE / "build" / "itch_book"
FLAGS = ["-std=c++17", "-O2", "-ffp-contract=off", "-fPIC", "-shared", "-Wall", "-Wextra", "-Wpedantic"]


def compiler() -> str:
    return os.environ.get("CXX", "c++")


def build() -> Path:
    LIBRARY.parent.mkdir(exist_ok=True)
    subprocess.run([compiler(), *FLAGS, str(SOURCE), "-o", str(LIBRARY)], check=True)
    return LIBRARY


def build_itch() -> Path:
    ITCH_BINARY.parent.mkdir(exist_ok=True)
    flags = [f for f in FLAGS if f not in ("-fPIC", "-shared")]
    subprocess.run([compiler(), *flags, str(ITCH_SOURCE), "-o", str(ITCH_BINARY)], check=True)
    return ITCH_BINARY


if __name__ == "__main__":
    try:
        print(build())
        print(build_itch())
    except (OSError, subprocess.CalledProcessError) as error:
        sys.exit(f"native build failed: {error}")
