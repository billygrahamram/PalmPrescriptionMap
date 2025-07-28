#!/bin/bash

set -e  # Exit on error

# # Upgrade pip and install dependencies
# echo "📦 Installing Python dependencies..."
# pip install --upgrade pip
# pip install -r requirements.txt



# 🏎️ Enable max performance mode on Jetson
echo "⚙️ Running power optimization script..."
python power.py




# Check Docker
if ! which docker &> /dev/null; then
    echo "❌ Docker is not installed or not in PATH."
    exit 1
fi
echo "🛠 Docker found: $(which docker)"

# Pull ODM if needed
if ! docker image inspect opendronemap/odm &> /dev/null; then
    echo "📦 Pulling OpenDroneMap image..."
    docker pull opendronemap/odm
fi

# Build your image
echo "🐳 Building palm-prescription-map..."
docker build -t palm-prescription-map .

# Run your container and capture the project path from output
echo "🚀 Running palm-prescription-map..."


docker run --rm -it \
    -v /media:/media \
    -v /run/media:/run/media \
    -v "$HOME/Downloads":"$HOME/Downloads" \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$(pwd)":"$(pwd)" \
    -w "$(pwd)" \
    palm-prescription-map



# Run ODM
docker run -ti --rm -v $(pwd)/output/project_1/datasets:/datasets opendronemap/odm --project-path /datasets project --fast-orthophoto --orthophoto-png --orthophoto-resolution 0.001

python inference.py

