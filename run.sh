#!/bin/bash
PYTHON="/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$PYTHON" "$SCRIPT_DIR/clicky.py"
