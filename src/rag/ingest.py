"""
Parses the kb/*.md documents.
Extracts metadata (doc_type, status, effective_from) and splits by SECTION-ID.
"""
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List

from langchain_text_splitters import MarkdownHeaderTextSplitter
from src.config import PATHS

@dataclass
class KBChunk:
    text: str
    section_id: str
    doc_id: str
    doc_type: str
    status: str
    effective_from: str | None

def parse_header(header_block: str) -> dict:
    """Parses the simple key: value metadata block at the top of each md file."""
    meta = {}
    for line in header_block.split("\n"):
        if ":" in line:
            key, val = line.split(":", 1)
            meta[key.strip()] = val.strip()
    return meta

def process_kb_files() -> List[KBChunk]:
    """Read all markdown files and return a list of chunks."""
    chunks = []
    kb_path = PATHS.kb
    
    if not kb_path.exists():
        raise FileNotFoundError(f"KB directory not found: {kb_path}")

    for md_file in kb_path.glob("*.md"):
        raw_text = md_file.read_text(encoding="utf-8")
        
        # 1. Extract document-level metadata (between the first pair of '---')
        header_match = re.match(r"^---\n(.*?)\n---\n", raw_text, re.DOTALL)
        doc_meta = {}
        if header_match:
            doc_meta = parse_header(header_match.group(1))
            # Remove the header from the text so it doesn't get chunked
            raw_text = raw_text[header_match.end():]
        
        doc_type = doc_meta.get("doc_type", "unknown")
        status = doc_meta.get("status", "current")
        effective_from = doc_meta.get("effective_from")
        doc_id = doc_meta.get("doc_id", md_file.stem)

        # 2. Split by Markdown sections (## [SECTION-ID] Title)
        # We don't use the splitter here because we specifically need the [SECTION-ID]
        # regex matches: ## [RET-2] Return windows
        section_pattern = re.compile(r"^##\s*\[(.*?)\]\s*(.*)$", re.MULTILINE)
        
        # Find all section starts
        matches = list(section_pattern.finditer(raw_text))
        
        for i, match in enumerate(matches):
            section_id = match.group(1)
            title = match.group(2)
            
            # Text is from this header to the next header (or EOF)
            start_pos = match.start()
            end_pos = matches[i+1].start() if i+1 < len(matches) else len(raw_text)
            
            chunk_text = raw_text[start_pos:end_pos].strip()
            
            # Only keep if there is actual content beyond the title
            if len(chunk_text) > len(match.group(0)):
                chunks.append(KBChunk(
                    text=chunk_text,
                    section_id=section_id,
                    doc_id=doc_id,
                    doc_type=doc_type,
                    status=status,
                    effective_from=effective_from
                ))
                
    return chunks