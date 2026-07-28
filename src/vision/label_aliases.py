"""Aliases de exibição para labels do dataset YOLO (sem retreinar)."""

from __future__ import annotations

# Nomes do data.yaml → leitura humana (UI / relatório).
# O modelo continua prevendo o label original; isto só renomeia na apresentação
# e no texto enviado ao LLM (prompt), sem alterar o vocabulário do peso.
LABEL_DISPLAY_ALIASES: dict[str, str] = {
    "sass_services": "saas_services",
    "aws_elactic_file_system(nfs)_multi-az": "aws_elastic_file_system_nfs_multi_az",
}


def display_label(name: str) -> str:
    """Devolve nome legível; desconhecido permanece igual."""
    key = str(name).strip()
    return LABEL_DISPLAY_ALIASES.get(key, key)


def display_labels(names: list[str]) -> list[str]:
    return [display_label(n) for n in names]
