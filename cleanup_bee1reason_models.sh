#!/bin/bash

# Script to clean up unnecessary blobs for Bee1reason models
# This script removes unnecessary blobs while keeping the essential ones

echo "Starting cleanup of Bee1reason Arabic models cache..."

# 1. Remove the entire beetleware/Bee1reason-arabic-Qwen-14B-Q4_K_M-GGUF directory
BEETLEWARE_DIR="$HOME/.cache/huggingface/hub/models--beetleware--Bee1reason-arabic-Qwen-14B-Q4_K_M-GGUF"

# Remove the entire directory
echo "Removing entire beetleware model directory..."
if [[ -d "$BEETLEWARE_DIR" ]]; then
    # Get directory size before removal
    SIZE=$(du -sh "$BEETLEWARE_DIR" | cut -f1)
    echo "Removing $BEETLEWARE_DIR ($SIZE)"
    rm -rf "$BEETLEWARE_DIR"
    echo "Removed the entire beetleware model directory"
else
    echo "Beetleware model directory not found"
fi

# 2. Cleanup mradermacher/Bee1reason-arabic-Qwen-14B-i1-GGUF
# Keep only the i1-Q4_K_S and i1-Q4_K_M blobs
MRADERMACHER_DIR="$HOME/.cache/huggingface/hub/models--mradermacher--Bee1reason-arabic-Qwen-14B-i1-GGUF/blobs"

# Blobs to keep - i1-Q4_K_S and i1-Q4_K_M
Q4KS_BLOB="334c5a6e68b4b18828cc3aeb771a22bd9bcbbca6d55c29ce507f3967c7db1af5"  # i1-Q4_K_S blob
Q4KM_BLOB="b17a5d0aa671fe8fad0abdc9c9914386f9a5b73bc69366500ed8f99b371a4552"  # i1-Q4_K_M blob

echo -e "\nCleaning mradermacher model directory..."
if [[ -d "$MRADERMACHER_DIR" ]]; then
    # Keep track of removed size
    TOTAL_SIZE_REMOVED_2=0
    
    for blob in "$MRADERMACHER_DIR"/*; do
        blob_name=$(basename "$blob")
        # Skip small files (likely configs) and our keep blobs
        if [[ "$blob_name" == "$Q4KS_BLOB" || "$blob_name" == "$Q4KM_BLOB" || $(stat -c%s "$blob") -lt 1000000 ]]; then
            echo "Keeping $blob_name"
        else
            SIZE=$(du -b "$blob" | cut -f1)
            echo "Removing $blob_name ($(numfmt --to=iec $SIZE))"
            rm "$blob"
            TOTAL_SIZE_REMOVED_2=$((TOTAL_SIZE_REMOVED_2 + SIZE))
        fi
    done
    
    echo "Freed $(numfmt --to=iec $TOTAL_SIZE_REMOVED_2) from mradermacher model"
else
    echo "Mradermacher model directory not found"
fi

# Summary
echo -e "\nCleanup completed!"
echo "Bee1reason models cleanup summary:"
echo "1. Removed entire beetleware/Bee1reason-arabic-Qwen-14B-Q4_K_M-GGUF directory"
echo "2. mradermacher model: Kept blobs:"
echo "   - $Q4KS_BLOB (i1-Q4_K_S)"
echo "   - $Q4KM_BLOB (i1-Q4_K_M)"
echo "   - Removed all other blobs (orphaned blobs and Q8 quantizations)"
