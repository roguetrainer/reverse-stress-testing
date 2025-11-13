#!/bin/bash

# Setup script for Reverse Stress Testing for Supply Chain Resilience
# Creates virtual environment and installs all dependencies

set -e  # Exit on error

echo "=========================================================================="
echo "Reverse Stress Testing - Environment Setup"
echo "=========================================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${BLUE}Checking Python installation...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✓${NC} Found Python $PYTHON_VERSION"
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✓${NC} Found Python $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version is 3.8+
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo -e "${RED}✗${NC} Python 3.8 or higher is required. Found $PYTHON_VERSION"
    exit 1
fi

echo ""

# Create virtual environment
echo -e "${BLUE}Creating virtual environment...${NC}"
if [ -d "venv" ]; then
    echo -e "${BLUE}Virtual environment already exists. Removing old one...${NC}"
    rm -rf venv
fi

$PYTHON_CMD -m venv venv
echo -e "${GREEN}✓${NC} Virtual environment created"
echo ""

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓${NC} Virtual environment activated"
echo ""

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo -e "${GREEN}✓${NC} pip upgraded"
echo ""

# Install dependencies
echo -e "${BLUE}Installing dependencies from requirements.txt...${NC}"
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}✗${NC} requirements.txt not found!"
    exit 1
fi

echo "  - numpy"
echo "  - pandas"
echo "  - scipy"
echo "  - networkx"
echo "  - matplotlib"
echo "  - seaborn"
echo ""

pip install -r requirements.txt

echo ""
echo -e "${GREEN}✓${NC} All dependencies installed successfully"
echo ""

# Test imports
echo -e "${BLUE}Testing installation...${NC}"
python -c "
import numpy as np
import pandas as pd
import scipy
import networkx as nx
import matplotlib
import seaborn as sns
print('All packages imported successfully!')
" && echo -e "${GREEN}✓${NC} Installation test passed" || echo -e "${RED}✗${NC} Installation test failed"

echo ""
echo "=========================================================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the virtual environment:"
echo -e "   ${BLUE}source venv/bin/activate${NC}"
echo ""
echo "2. Run the demonstration:"
echo -e "   ${BLUE}python demo_rst.py${NC}"
echo ""
echo "3. Or test core functionality:"
echo -e "   ${BLUE}python reverse_stress_testing.py${NC}"
echo ""
echo "4. To deactivate the virtual environment later:"
echo -e "   ${BLUE}deactivate${NC}"
echo ""
echo "=========================================================================="
echo ""
echo "Documentation:"
echo "  - Start here: INDEX.md"
echo "  - Quick start: QUICKSTART.md"
echo "  - Full docs: README.md"
echo ""
echo "=========================================================================="
