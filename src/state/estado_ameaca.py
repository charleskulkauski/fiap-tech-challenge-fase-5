from typing import Annotated, Any, Dict, List, TypedDict
import operator


class EstadoAmeacas(TypedDict, total=False):
    """Estado compartilhado do pipeline LangGraph."""

    caminho_imagem: str
    componentes_detectados: List[str]
    deteccoes: List[Dict[str, Any]]
    imagem_anotada: str
    analise_stride: Dict[str, Any]
    vulnerabilidades: List[Dict[str, Any]]
    etapa_atual: str
    relatorio_final: str
    avisos: Annotated[List[str], operator.add]
