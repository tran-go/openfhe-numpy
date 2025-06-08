#!/bin/bash

# Prompt for commit message
read -p "Enter commit message: " COMMIT_MSG

# Validate commit message
if [ -z "$COMMIT_MSG" ]; then
    echo "❌ Commit message cannot be empty."
    exit 1
fi

# Prompt for mode
echo "Choose an option:"
echo "1) Commit all files in core/ and openfhe_numpy/"
echo "2) Commit hardcoded list (can include *.cpp, *.py)"
read -p "Enter 1 or 2: " CHOICE

# Option 1: commit everything in folders
if [ "$CHOICE" = "1" ]; then
    echo "➤ Staging all files in core/ and openfhe_numpy/..."

    [ -d "core" ] && git add -A core || echo "⚠️ Directory 'core' not found."
    [ -d "openfhe_numpy" ] && git add -A openfhe_numpy || echo "⚠️ Directory 'openfhe_numpy' not found."

# Option 2: hardcoded list with globs
elif [ "$CHOICE" = "2" ]; then
    echo "➤ Staging files from hardcoded list..."

    FILE_PATTERNS=(
        "docs/"
        "examples/*.py"
        "tests/*.py"
        "tests/ckks_params.csv"
        "core/examples/*.cpp"
        "core/include/openfhe_numpy/*.h"
        "core/src/*.cpp"
        "openfhe_numpy/operations/*.py"
        "openfhe_numpy/tensor/*.py"
        "openfhe_numpy/utils/*.py"
        "openfhe_numpy/*.py"
        "openfhe_numpy/version.py.in"
        "LICENSE"
        "README.md"
        "CMakeLists.txt"
        "install.sh"
    )

    for pattern in "${FILE_PATTERNS[@]}"; do
        echo "Processing pattern: $pattern"
        # Use git ls-files for existing files and git ls-files --others for untracked files
        files=$(git ls-files "$pattern" 2>/dev/null; git ls-files --others --exclude-standard "$pattern" 2>/dev/null)
        
        if [ -z "$files" ]; then
            echo "⚠️  No files match pattern: $pattern"
        else
            for file in $files; do
                if [ -f "$file" ]; then
                    git add "$file"
                    echo "✅ Added: $file"
                fi
            done
        fi
    done

else
    echo "Invalid option. Exiting!!!"
    exit 1
fi

# Show git status before proceeding
echo
echo "🔍 Git status preview:"
git status

# Ask for confirmation
read -p "Proceed with commit and push? (y/n): " CONFIRM
if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    echo "Aborted!!!"
    exit 0
fi

# Commit and push
if git commit -m "$COMMIT_MSG"; then
    echo "Commit successful"
    
    # Try to push, handle potential errors
    if git push; then
        echo "Push successful"
    else
        echo "Push failed. You may need to set up tracking or resolve conflicts."
        echo "Try: git push --set-upstream origin $(git branch --show-current)"
    fi
else
    echo "Commit failed!!!"
    exit 1
fi