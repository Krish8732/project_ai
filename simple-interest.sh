#!/bin/bash
# Simple interest calculator
# Usage: bash simple-interest.sh principal rate time
PRINCIPAL=$1
RATE=$2
TIME=$3
INTEREST=$(echo "$PRINCIPAL * $RATE * $TIME / 100" | bc)
echo "Simple Interest = $INTEREST"
