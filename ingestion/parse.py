"""Парсинг seed-документов и разбиение на чанки с метаданными ACL."""
from pathlib import Path
from typing import Iterator


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Абзацно-ориентированное чанкование с перекрытием."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) > chunk_size and current:
            chunks.append(current.strip())
            current = current[-overlap:] if overlap else ""
        current += ("\n\n" if current else "") + p
    if current.strip():
        chunks.append(current.strip())
    return chunks


def iter_documents(seed_dir: str) -> Iterator[dict]:
    """
    Обходит data/seed/<group>/<file> и отдаёт чанки с ACL.
    Имя папки это группа доступа, для 'public' выставляется ['all'].
    """
    root = Path(seed_dir)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        group = path.relative_to(root).parts[0]
        allowed_groups = ["all"] if group == "public" else [group]

        text = path.read_text(encoding="utf-8")
        for idx, chunk in enumerate(chunk_text(text)):
            yield {
                # группа в id, чтобы одноимённые файлы в разных папках не затирали друг друга
                "id": f"{group}/{path.stem}-{idx}",
                "text": chunk,
                "source": path.relative_to(root).as_posix(),
                "group": group,
                "allowed_groups": allowed_groups,
                "chunk_index": idx,
            }