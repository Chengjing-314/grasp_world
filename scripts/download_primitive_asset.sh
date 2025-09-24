#!/usr/bin/env bash
# scripts/download_primitive_shapes.sh
# Download primitive shape meshes into project_root/assets/primitive_shapes/

set -e

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

TARGET_DIR="$PROJECT_ROOT/assets/primitive_shapes"
mkdir -p "$TARGET_DIR"

declare -A FILES=(
  ["sphere.obj"]="https://web.mit.edu/djwendel/www/weblogo/shapes/basic-shapes/sphere/sphere.obj"
  ["cube.obj"]="https://web.mit.edu/djwendel/www/weblogo/shapes/basic-shapes/cube/cube.obj"
  ["cylinder.obj"]="https://web.mit.edu/djwendel/www/weblogo/shapes/basic-shapes/cylinder/cylinder.obj"
  ["cone.obj"]="https://web.mit.edu/djwendel/www/weblogo/shapes/basic-shapes/cone/cone.obj"
)

for FILE in "${!FILES[@]}"; do
  URL="${FILES[$FILE]}"
  DEST="$TARGET_DIR/$FILE"

  if [ -f "$DEST" ]; then
    echo "[skip] $FILE already exists at $DEST"
  else
    echo "[download] $FILE -> $DEST"
    curl -sS -L -o "$DEST" "$URL"
  fi
done

echo "All done! Files are in $TARGET_DIR"
