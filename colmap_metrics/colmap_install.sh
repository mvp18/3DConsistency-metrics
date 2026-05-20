#!/bin/bash

# ==============================================================================
# Universal COLMAP Installer (via Docker)
# Tested on: Arch Linux / Manjaro
# Compatible with: Ubuntu, Debian, Fedora (requires manual Docker install first)
# ==============================================================================

set -e # Exit on error

echo -e "\n======================================================="
echo "  COLMAP Docker Setup & Alias Installer"
echo -e "=======================================================\n"

# 1. OS & Dependency Check
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "Detected OS: $PRETTY_NAME"
fi

echo -e "\n--- Checking Dependencies ---"
MISSING_DEPS=0

if ! command -v docker &> /dev/null; then
    echo "[!] Docker is not installed."
    MISSING_DEPS=1
fi

if ! command -v nvidia-ctk &> /dev/null; then
    echo "[!] NVIDIA Container Toolkit is not installed."
    MISSING_DEPS=1
fi

# 2. Auto-Install for Arch/Manjaro
if [ $MISSING_DEPS -eq 1 ]; then
    if [[ "$ID" == "arch" || "$ID_LIKE" == *"arch"* || "$ID" == "manjaro" ]]; then
        echo "[+] Arch/Manjaro detected. Auto-installing Docker and NVIDIA toolkit..."
        sudo pacman -S --needed docker nvidia-container-toolkit
    else
        echo -e "\n[!] ERROR: Please install 'docker' and 'nvidia-container-toolkit' for your specific distribution first."
        echo "Ubuntu/Debian: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        exit 1
    fi
fi

# 3. Configure Docker & NVIDIA Runtime
echo -e "\n--- Configuring Docker for NVIDIA GPUs ---"
sudo systemctl enable --now docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 4. Pull COLMAP Image
echo -e "\n--- Pulling Official COLMAP Docker Image ---"
sudo docker pull colmap/colmap:latest

# 5. Inject Terminal Alias
echo -e "\n--- Setting up native terminal alias ---"

ALIAS_BLOCK='
# ==========================================
# COLMAP Docker Alias
# ==========================================
function colmap_docker() {
    sudo docker run --rm --gpus all \
    -v "$(pwd)":"$(pwd)" \
    -w "$(pwd)" \
    colmap/colmap:latest colmap "$@"
}
alias colmap=colmap_docker
'

install_alias() {
    local rc_file="$1"
    if [ -f "$rc_file" ]; then
        if grep -q "colmap_docker" "$rc_file"; then
            echo "[✓] Alias already exists in $rc_file"
        else
            echo "$ALIAS_BLOCK" >> "$rc_file"
            echo "[+] Alias added to $rc_file"
        fi
    fi
}

# Add to both common shells
install_alias "$HOME/.bashrc"
install_alias "$HOME/.zshrc"

echo -e "\n======================================================="
echo "  INSTALLATION COMPLETE!"
echo "======================================================="
echo -e "To use COLMAP immediately, reload your shell by running:"
echo -e "  source ~/.bashrc   (or source ~/.zshrc)\n"
echo -e "Then test it with:"
echo -e "  colmap help"
echo -e "=======================================================\n"