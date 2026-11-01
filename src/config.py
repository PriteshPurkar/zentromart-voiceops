"""
Central configuration for ZentroMart VoiceOps 2026.
Loads .env, exposes paths, constants, and provider switches.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------
# Project root = parent of src/
# ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

# Load .env from project root (no-op if missing)
load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------
# The dataset's fixed "today" — 20 October 2026.
# Use this EVERYWHERE instead of date.today().
# ---------------------------------------------------------------
TODAY: date = date(2026, 10, 20)


# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    data: Path = ROOT / os.getenv("DATA_DIR", "data")
    kb: Path = ROOT / os.getenv("KB_DIR", "kb")
    practice: Path = ROOT / os.getenv("PRACTICE_DIR", "practice")
    tools: Path = ROOT / os.getenv("TOOLS_DIR", "tools")
    index: Path = ROOT / os.getenv("INDEX_DIR", "cache/index")
    outputs: Path = ROOT / os.getenv("OUTPUT_DIR", "sample_outputs")

    def ensure(self) -> "Paths":
        """Create output/index dirs if they don't exist."""
        for p in (self.index, self.outputs):
            p.mkdir(parents=True, exist_ok=True)
        return self


PATHS = Paths().ensure()


# ---------------------------------------------------------------
# LLM settings
# ---------------------------------------------------------------
PROVIDER: str = os.getenv("LLM_PROVIDER", "groq").lower()

OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# ---------------------------------------------------------------
# Embeddings / RAG
# ---------------------------------------------------------------
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIM: int = 384  # MiniLM-L6-v2

# Retrieval
TOP_K: int = 6
# Precedence order per PRD: current policy > agent guide > help > faq > marketing
DOC_TYPE_PRECEDENCE: tuple[str, ...] = (
    "policy", "agent_guide", "help", "faq", "marketing", "meta",
)


# ---------------------------------------------------------------
# Gnani voice
# ---------------------------------------------------------------
GNANI_API_KEY: str | None = os.getenv("GNANI_API_KEY")
GNANI_STT_LANGUAGE: str = os.getenv("GNANI_STT_LANGUAGE", "en-IN")
GNANI_TTS_VOICE: str = os.getenv("GNANI_TTS_VOICE", "Kaveri")
GNANI_TTS_MODEL: str = os.getenv("GNANI_TTS_MODEL", "timbre-v2.5")
GNANI_TTS_LANGUAGE: str = os.getenv("GNANI_TTS_LANGUAGE", "en-IN")


# ---------------------------------------------------------------
# Business-day rules (README: Mon–Sat, skip holidays.csv)
# ---------------------------------------------------------------
BUSINESS_WEEKDAYS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)  # Mon=0 … Sat=5


# ---------------------------------------------------------------
# Sanity check — call this once at startup
# ---------------------------------------------------------------
def validate() -> list[str]:
    """Return a list of warnings (empty = all good)."""
    warnings: list[str] = []

    for name, p in [
        ("data", PATHS.data),
        ("kb", PATHS.kb),
        ("practice", PATHS.practice),
        ("tools", PATHS.tools),
    ]:
        if not p.exists():
            warnings.append(f"missing directory: {name} -> {p}")

    if PROVIDER == "openai" and not OPENAI_API_KEY:
        warnings.append("LLM_PROVIDER=openai but OPENAI_API_KEY is empty")
    if PROVIDER == "groq" and not GROQ_API_KEY:
        warnings.append("LLM_PROVIDER=groq but GROQ_API_KEY is empty")

    if not GNANI_API_KEY:
        warnings.append("GNANI_API_KEY is empty (voice step will fail)")

    return warnings


if __name__ == "__main__":
    print(f"ROOT        = {ROOT}")
    print(f"TODAY       = {TODAY}  (fixed dataset reference date)")
    print(f"PROVIDER    = {PROVIDER}")
    print(f"DATA exists = {PATHS.data.exists()}")
    print(f"KB   exists = {PATHS.kb.exists()}")
    print(f"TOOLS exists= {PATHS.tools.exists()}")
    warns = validate()
    if warns:
        print("\nWarnings:")
        for w in warns:
            print(f"  - {w}")
    else:
        print("\nNo warnings — config OK.")