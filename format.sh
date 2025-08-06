#!/bin/bash
# Script to format all code in the project

echo "🔧 Starting code formatting..."
echo ""

# Format Python files with Ruff
echo "📐 Formatting Python files with Ruff..."
uv run ruff format .
if [ $? -eq 0 ]; then
    echo "✅ Python formatting completed"
else
    echo "❌ Python formatting failed"
fi

echo ""
echo "🔍 Checking and fixing Python linting issues..."
uv run ruff check . --fix
if [ $? -eq 0 ]; then
    echo "✅ Python linting fixes completed"
else
    echo "⚠️  Some Python linting issues remain (manual fix required)"
fi

echo ""
echo "📋 Checking YAML files..."
uv run yamllint -f parsable .
if [ $? -eq 0 ]; then
    echo "✅ All YAML files are properly formatted"
else
    echo "⚠️  Some YAML files have formatting issues"
fi

echo ""
echo "🎉 Formatting complete!"
