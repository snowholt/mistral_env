#!/bin/bash

# Script to remove unnecessary quantizations from unsloth Qwen3-14B-GGUF cache
# Keep only Q4_K_S, Q4_K_M, Q6_K, Q8_0

UNSLOTH_BLOBS_DIR="$HOME/.cache/huggingface/hub/models--unsloth--Qwen3-14B-GGUF/blobs"

# Blobs to KEEP (our desired quantizations)
KEEP_BLOBS=(
    "25b653915ea21da7a0983b3804be167b7e205646b87af8a3daa1d8bf2de46111"  # Q4_K_S (8.0GB)
    "5eaa0870bd81ed3b58a630a271234cfa604e43ffb3a19cd68e54a80dd9d52a66"   # Q4_K_M (8.4GB)
    "c34d749069d5f19b998498ce84884975c551529548fa6a56b883345d166289c2"   # Q6_K (12GB)
    "90224247d4a8076c0a689e833910f4291bce05dec81f472ebcba321607168ea1"  # Q8_0 (15GB)
)

# Large blobs to REMOVE (unnecessary quantizations)
REMOVE_BLOBS=(
    "6d47b3bb874039b1e0c26e7a7dacc5268833a601325095d97559a70c4de1df99"   # 3.6G
    "469f8b1e8f0ae192768eb14c9fceed0a07f63b44320c9b23c230b5ed6b7c8e79"   # 3.8G 
    "d8d42e06ea9fea7662b03a1c3cdbcd31c246c1017aa43b0366844bf2b9319983"   # 4.2G
    "3c4aa2f82f9f9c9b47ebbda4dc384d7a0fa3031f4d0ae4241e0fa6420aa0d9b1"   # 5.1G
    "7282a1791062c250b306c8b8c97c77cd130046898d3ac4c1ab3ccaa76fa721f0"   # 5.4G
    "fa3df90613fb148bb41fda4dc24b3648fff9de3d32c1b837d967dc9e628c268a"   # 5.6G
    "346f4dcf6bd85dfc6145eeb0da8efdfacae879cb9b0c8fe5cad85e0ae0ae80ce"   # 5.7G
    "6abefcfb88de4157acc6109a711af65fdb5a6da2a77a1ac91bdada2ded22fd49"   # 5.7G
    "b5dad229e5b047f2627bd90bc0ab1e6dfc02510b904aa1b13d1e152e067d8736"   # 6.2G
    "9694f1db203d5a18bea6f19a36be01b2d2c79bf9b7f8df553a4cc5cff870b47e"   # 6.9G
    "46b8c8a3fda3de374f465be47176c956e44c7ef56ceddf6aa8bafddb1c73d5b6"   # 7.1G
    "e24bdda1295a1bb7898b7af9300b86eba0ffba66d8f3dadbc90d00e98ecd5369"   # 7.6G
    "e3d692d9e63af04bd212f8b4a1dcc6a49a0fce094343f0757ed0fd0b25473c39"   # 8.0G (duplicate)
    "009f54ffc8d8082e7921139924229d4deea61c9174a0a357d91384bd299ff78e"   # 8.0G (duplicate)
    "7f371adeaaf51dc97da927c61e6573e75e3ea33d76ec09dddb8cad00d0d5b6cb"   # 8.6G
    "e855b72109508f8cf4bef272ee255a41c56ec8765eee87351febcff18fb93736"   # 8.8G
    "231f2cea7c4ffc2748cb5556ec7190eeef4e19a900d8ae55549d301d7518d26b"   # 9.6G
    "305309039ea44df963c01f66d65e949f8eebca238554916df437f02295a3a413"   # 9.8G
    "f62da26805a70d9963400ca6160843c82e1aaadc714da8b299d0a676676741b8"   # 9.9G
    "83e7dcec42952a4ed6e092977c587a9ab8d528d52d0db481df6d694d5181811b"   # 13G
    "2da4750f82103624cbc6f831a5907628670c9b2ba59c4d57e59186c19860f4b3"   # 18G
    "1d12adb9b56bea81a1b3b8b2eea0fcf5700307ba4ed5d56d2f64f222ec2d7b72"   # 28G
)

echo "Starting cleanup of unnecessary Qwen3 quantizations from unsloth repository..."
echo "Will keep: Q4_K_S, Q4_K_M, Q6_K, Q8_0"
echo

# Calculate space to be freed
TOTAL_SIZE=0
for blob in "${REMOVE_BLOBS[@]}"; do
    if [[ -f "$UNSLOTH_BLOBS_DIR/$blob" ]]; then
        SIZE=$(du -b "$UNSLOTH_BLOBS_DIR/$blob" | cut -f1)
        TOTAL_SIZE=$((TOTAL_SIZE + SIZE))
    fi
done

echo "Estimated space to be freed: $(echo $TOTAL_SIZE | numfmt --to=iec)"
echo

# Remove unnecessary blobs
removed_count=0
for blob in "${REMOVE_BLOBS[@]}"; do
    if [[ -f "$UNSLOTH_BLOBS_DIR/$blob" ]]; then
        size=$(du -h "$UNSLOTH_BLOBS_DIR/$blob" | cut -f1)
        echo "Removing blob $blob ($size)..."
        rm "$UNSLOTH_BLOBS_DIR/$blob"
        ((removed_count++))
    else
        echo "Blob $blob not found (already removed?)"
    fi
done

echo
echo "Cleanup completed!"
echo "Removed $removed_count unnecessary quantization blobs"
echo "Keeping 4 optimal quantizations: Q4_K_S, Q4_K_M, Q6_K, Q8_0"
