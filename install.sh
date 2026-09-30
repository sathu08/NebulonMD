#!/usr/bin/env bash

# ============================================================
# NebulonMind Installation Script (curl | bash)
# Installs NebulonMind into a fixed location (~/.nebulonmind)
# and creates a global 'nebulonmind' launcher in ~/.local/bin.
# ============================================================

set -euo pipefail

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

REPO_URL="https://github.com/sathu08/NebulonMD.git"
BRANCH="main"

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------

log() {
    echo "[NebulonMind] $*"
}

error() {
    echo "[NebulonMind][ERROR] $*" >&2
    exit 1
}

# ------------------------------------------------------------
# Check Git
# ------------------------------------------------------------

if ! command -v git >/dev/null 2>&1; then
    error "Git is not installed. Please install Git first."
fi

# ------------------------------------------------------------
# Clone or Update NebulonMind
# ------------------------------------------------------------

log "NebulonMind repository:"
log "$REPO_URL"

log "Target branch:"
log "$BRANCH"

PROJECT_DIR="$HOME/.nebulonmind"

log "Install directory:"
log "$PROJECT_DIR"

if [[ -d "$PROJECT_DIR/.git" ]]; then

    # ---------- Existing repository: fetch + update ----------

    CURRENT_REMOTE="$(git remote get-url origin 2>/dev/null || true)"

    if [[ -z "$CURRENT_REMOTE" ]]; then
        error "Git remote 'origin' is not configured in $PROJECT_DIR"
    fi

    log "Git remote:"
    log "$CURRENT_REMOTE"

    log "Fetching latest changes from branch '$BRANCH'..."

    git fetch origin "$BRANCH"

    log "Switching to branch '$BRANCH'..."

    if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
        git checkout "$BRANCH"
    else
        git checkout -b "$BRANCH" --track "origin/$BRANCH"
    fi

    log "Updating branch '$BRANCH'..."

    git pull --ff-only origin "$BRANCH"

else

    # ---------- Fresh clone ----------

    if [[ -d "$PROJECT_DIR" ]]; then
        error "NebulonMind directory already exists but is not a Git repository: $PROJECT_DIR"
    fi

    log "Cloning NebulonMind into:"
    log "$PROJECT_DIR"
    log "Branch: $BRANCH"

    git clone \
        --branch "$BRANCH" \
        --single-branch \
        "$REPO_URL" \
        "$PROJECT_DIR"

    log "Repository cloned successfully."
fi

# ------------------------------------------------------------
# Set Current Directory to NebulonMind
# ------------------------------------------------------------

cd "$PROJECT_DIR"

log "NebulonMind Home:"
log "$PROJECT_DIR"

# ------------------------------------------------------------
# Verify Branch
# ------------------------------------------------------------

CURRENT_BRANCH="$(git branch --show-current)"

if [[ "$CURRENT_BRANCH" != "$BRANCH" ]]; then
    error "Expected branch '$BRANCH', but currently on '$CURRENT_BRANCH'."
fi

log "Git branch:"
log "$CURRENT_BRANCH"

# ------------------------------------------------------------
# Check / Install uv
# ------------------------------------------------------------

if ! command -v uv >/dev/null 2>&1; then
    log "uv is not installed."
    log "Installing uv..."

    curl -LsSf https://astral.sh/uv/install.sh | sh

    export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
    error "uv installation failed or uv is not available in PATH."
fi

log "uv version:"
uv --version

# ------------------------------------------------------------
# Check / Install Python 3.10
# ------------------------------------------------------------

PYTHON_VERSION="3.10"

log "Checking Python $PYTHON_VERSION..."

if uv python find "$PYTHON_VERSION" >/dev/null 2>&1; then
    log "Python $PYTHON_VERSION is already installed."
else
    log "Python $PYTHON_VERSION not found."
    log "Installing Python $PYTHON_VERSION..."

    uv python install "$PYTHON_VERSION"
fi

PYTHON_PATH="$(uv python find "$PYTHON_VERSION")"

log "Using Python:"
log "$PYTHON_PATH"

# ------------------------------------------------------------
# Install NebulonMind
#
# Requires a running NebulonDB backend (default localhost:6969)
# at runtime; the install itself does not need one.
# ------------------------------------------------------------

log "Installing NebulonMind dependencies..."

uv sync --python "$PYTHON_VERSION"

# ------------------------------------------------------------
# Install Global NebulonMind CLI
# ------------------------------------------------------------

VENV_DIR="$PROJECT_DIR/.venv"
CLI_PATH="$VENV_DIR/bin/nebulonmind"
BIN_DIR="$HOME/.local/bin"
GLOBAL_CLI="$BIN_DIR/nebulonmind"

if [[ ! -x "$CLI_PATH" ]]; then
    error "NebulonMind CLI was not created: $CLI_PATH"
fi

log "Installing global NebulonMind CLI..."

mkdir -p "$BIN_DIR"

cat > "$GLOBAL_CLI" <<EOF
#!/usr/bin/env bash

export NEBULONMD_HOME="$PROJECT_DIR"

exec "$CLI_PATH" "\$@"
EOF

chmod +x "$GLOBAL_CLI"

# ------------------------------------------------------------
# Configure ~/.local/bin in PATH
# ------------------------------------------------------------

BASHRC="$HOME/.bashrc"

if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    export PATH="$HOME/.local/bin:$PATH"
fi

if [[ -f "$BASHRC" ]] && ! grep -qF 'export PATH="$HOME/.local/bin:$PATH"' "$BASHRC"; then
    printf '\n# User local binaries\nexport PATH="$HOME/.local/bin:$PATH"\n' >> "$BASHRC"
fi

# ------------------------------------------------------------
# Verify NebulonMind CLI
# ------------------------------------------------------------

if ! command -v nebulonmind >/dev/null 2>&1; then
    error "NebulonMind CLI was not installed correctly."
fi

log "NebulonMind executable:"
log "$(command -v nebulonmind)"

log "Testing NebulonMind CLI..."

nebulonmind --help || true

# ------------------------------------------------------------
# Installation Complete
# ------------------------------------------------------------

export NEBULONMD_HOME="$PROJECT_DIR"

echo ""
echo "============================================================"
echo " NebulonMind Installation Complete"
echo "============================================================"
echo ""
echo "Repository      : $REPO_URL"
echo "Branch          : $CURRENT_BRANCH"
echo "Project         : $PROJECT_DIR"
echo "Python          : $PYTHON_PATH"
echo "Virtual Env     : $VENV_DIR"
echo "NebulonMind CLI : $(command -v nebulonmind)"
echo "NEBULONMD_HOME  : $NEBULONMD_HOME"
echo ""
echo "Next steps:"
echo ""
echo "  1. Start NebulonDB first (backend):  nebulondb start"
echo "  2. Add NEBULONDB_USERNAME / NEBULONDB_PASSWORD to:"
echo "     $NEBULONMD_HOME/.env"
echo "  3. Open a new terminal (or: source ~/.bashrc), then:"
echo ""
echo "         nebulonmind start      # API on http://localhost:9696"
echo "         nebulonmind            # interactive chat TUI"
echo ""
echo "============================================================"
