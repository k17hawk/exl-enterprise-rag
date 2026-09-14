"""Walk the handbook corpus, filtered by the department allowlist."""

from __future__ import annotations


from src.exl_enterprise_rag.config.config_entity import SourceFile
from src.exl_enterprise_rag.config.settings import DepartmentConfig, Settings


def walk_corpus(
    settings: Settings,
    departments: DepartmentConfig,
    only_departments: set[str] | None = None,
) -> list[SourceFile]:
    """Return every .md file under an allowlisted folder.

    Args:
        settings: runtime settings (provides handbook_content_dir).
        departments: the allowlist config.
        only_departments: optional filter — if set, only files whose
            canonical department is in this set are returned.

    Skips:
        - any folder not present in folder_to_department
        - any file not ending in .md
        - any subdirectory starting with '.' or '_' (but keeps _index.md)
    """
    root = settings.handbook_content_dir
    if not root.is_dir():
        raise FileNotFoundError(f"handbook content dir not found: {root}")

    results: list[SourceFile] = []
    allowed_folders = departments.allowed_folders()

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith((".", "_")):
            continue
        if child.name not in allowed_folders:
            continue

        dept = departments.department_for(child.name)
        if dept is None:
            continue
        if only_departments is not None and dept not in only_departments:
            continue

        for md_path in sorted(child.rglob("*.md")):
            rel_parts = md_path.relative_to(child).parts
            # Skip files inside dot/underscore dirs, except _index.md itself
            if any(p.startswith(".") for p in rel_parts[:-1]):
                continue
            if any(p.startswith("_") and p != "_index.md" for p in rel_parts[:-1]):
                continue

            rel_to_handbook = md_path.relative_to(root)
            results.append(SourceFile(
                path=md_path,
                relative_path=str(rel_to_handbook),
                source_path=f"content/handbook/{rel_to_handbook}",
                folder=child.name,
                department=dept,
            ))

    return results