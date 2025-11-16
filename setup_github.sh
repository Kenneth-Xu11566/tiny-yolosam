#!/bin/bash
# Helper script to push this repository to GitHub
# Run after creating a new repository on GitHub

set -e

echo "========================================"
echo "GitHub Setup Helper"
echo "========================================"
echo ""

# Check if user has provided repo URL
if [ -z "$1" ]; then
    echo "Usage: ./setup_github.sh <github-repo-url>"
    echo ""
    echo "Steps:"
    echo "  1. Go to https://github.com/new"
    echo "  2. Create a new repository (e.g., 'yolo-tinysam-hybrid')"
    echo "  3. DO NOT initialize with README, .gitignore, or license"
    echo "  4. Copy the repository URL (e.g., https://github.com/YourUsername/yolo-tinysam-hybrid.git)"
    echo "  5. Run: ./setup_github.sh https://github.com/YourUsername/yolo-tinysam-hybrid.git"
    echo ""
    exit 1
fi

REPO_URL=$1

# Add remote
echo "Adding remote 'origin'..."
git remote add origin "$REPO_URL"

# Check current branch
BRANCH=$(git branch --show-current)
echo "Current branch: $BRANCH"

# Push to GitHub
echo ""
echo "Pushing to GitHub..."
git push -u origin "$BRANCH"

echo ""
echo "========================================"
echo "✅ Success! Repository pushed to GitHub"
echo "========================================"
echo ""
echo "View your repository at:"
echo "  ${REPO_URL%.git}"
echo ""
echo "Next steps:"
echo "  - Download model weights (see README.md)"
echo "  - Set up issues/projects on GitHub"
echo "  - Add collaborators if needed"
echo ""

