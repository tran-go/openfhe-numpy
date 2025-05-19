#!/bin/bash

# Prompt for commit message
read -p "Enter commit message: " COMMIT_MSG

# Prompt for mode
echo "Choose an option:"
echo "1) Commit all files in cpp/ and python/"
echo "2) Commit hardcoded list (can include *.cpp, *.py)"
read -p "Enter 1 or 2: " CHOICE

# Temporary file list
FILES_TO_ADD=()

# Option 1: commit everything in folders
if [ "$CHOICE" = "1" ]; then
    echo "➤ Staging all files in cpp/ and python/..."

    [ -d "cpp" ] && git add -A cpp || echo "Directory 'cpp' not found."
    [ -d "python" ] && git add -A python || echo "Directory 'python' not found."

# Option 2: hardcoded list with globs
elif [ "$CHOICE" = "2" ]; then
    echo "➤ Staging files from hardcoded list..."

    FILE_PATTERNS=(
        docs/
        examples/*.py
        tests/*.py
        tests/ckks_params.csv
        core/examples/simple-matrix-1-operations.cpp
        core/examples/simple-matrix-operations.cpp
        core/include/openfhe_numpy/*.h
        core/src/*.cpp
        openfhe_numpy/operations/*.py
        openfhe_numpy/tensor/tensor.py
        openfhe_numpy/tensor/ctarray.py
        openfhe_numpy/tensor/ptarray.py
        openfhe_numpy/tensor/constructors.py
        openfhe_numpy/tensor/__init__.py
        openfhe_numpy/util/*.py
        openfhe_numpy/config.py
        openfhe_numpy/__init__.py
        openfhe_numpy/version.py.in
        LICENSE
        README.md
        CMakeLists.txt)


    for pattern in "${FILE_PATTERNS[@]}"; do
        matches=($pattern)
        for file in "${matches[@]}"; do
            if [ -f "$file" ]; then
                git add "$file"
                FILES_TO_ADD+=("$file")
            else
                echo "⚠️  Not a file or doesn't exist: $file"
            fi
        done
    done

else
    echo "❌ Invalid option. Exiting."
    exit 1
fi

# Show git status before proceeding
echo
echo "🔍 Git status preview:"
git status

# Ask for confirmation
read -p "Proceed with commit and push? (y/n): " CONFIRM
if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    echo "❌ Aborted."
    exit 0
fi

# Commit and push
git commit -m "$COMMIT_MSG"
git push

