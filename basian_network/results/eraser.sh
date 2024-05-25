#!/bin/bash

# Check if the argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 {exm|est|exp}"
  exit 1
fi

# Set the target directory based on the argument
TARGET_DIR=""
case $1 in
  exm)
    TARGET_DIR="exm"
    ;;
  est)
    TARGET_DIR="est"
    ;;
  exp)
    TARGET_DIR="exp"
    ;;
  *)
    echo "Invalid argument. Use 'exm', 'est', or 'exp'."
    exit 1
    ;;
esac

# Remove all contents of the target directory
rm -rf "$TARGET_DIR"/*

# Recreate the required directories
mkdir -p "$TARGET_DIR/order"
mkdir -p "$TARGET_DIR/score"

echo "The contents of $TARGET_DIR have been erased and required directories have been created."
