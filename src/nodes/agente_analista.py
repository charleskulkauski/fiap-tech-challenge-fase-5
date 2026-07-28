"""Nó LangGraph: análise STRIDE e vulnerabilidades via Azure OpenAI."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")

from src.state.estado_ameaca import EstadoAmeacas  # noqa: E402

logger = logging.getLogger(__name__)

AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

# Nomes padrão Microsoft; legado OPENAPI_* ainda aceito (typo antigo do projeto).
_AZURE_KEY_ENVS = ("AZURE_OPENAI_API_KEY", "AZURE_OPENAPI_KEY")
_AZURE_ENDPOINT_ENVS = ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAPI_BASE_URL")

SYSTEM_PROMPT = """\
Você é um especialista em segurança da informação e modelagem de ameaças (STRIDE).
Analise componentes de um diagrama de arquitetura de software/nuvem.

Categorias STRIDE:
- S (Spoofing): falsificação de identidade
- T (Tampering): adulteração de dados/código
- R (Repudiation): negação de ações sem auditoria
- I (Information Disclosure): vazamento de informação
- D (Denial of Service): indisponibilidade
- E (Elevation of Privilege): escalação de privilégios

Regras:
- Responda APENAS com JSON válido (sem markdown, sem texto fora do JSON).
- Use riscos genéricos e controles OWASP/cloud — NÃO invente CVEs específicos.
- Priorize ameaças realistas para cada tipo de componente.
- Inclua contramedidas práticas e acionáveis.
- Se houver posições (x, y), use-as para inferir fluxo aproximado
  (ex.: usuário à esquerda → gateway → backend à direita).
"""

USER_PROMPT_TEMPLATE = """\
Componentes detectados no diagrama:
{componentes}

Detecções com posição (quando disponíveis):
{deteccoes}

Retorne JSON neste formato exato:
{{
  "fluxo_inferido": "descrição curta do fluxo entre componentes",
  "analise_stride": {{
    "<nome_do_componente>": {{
      "S": {{"ameaca": "...", "contramedida": "..."}},
      "T": {{"ameaca": "...", "contramedida": "..."}},
      "R": {{"ameaca": "...", "contramedida": "..."}},
      "I": {{"ameaca": "...", "contramedida": "..."}},
      "D": {{"ameaca": "...", "contramedida": "..."}},
      "E": {{"ameaca": "...", "contramedida": "..."}}
    }}
  }},
  "vulnerabilidades": [
    {{
      "componente": "...",
      "risco": "...",
      "severidade": "alta|media|baixa",
      "categoria_stride": "S|T|R|I|D|E",
      "contramedida": "..."
    }}
  ]
}}
"""


def _get_env_first(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _require_azure_env() -> tuple[str, str]:
    api_key = _get_env_first(*_AZURE_KEY_ENVS)
    base_url = _get_env_first(*_AZURE_ENDPOINT_ENVS)
    if not api_key or not base_url:
        raise RuntimeError(
            "Defina AZURE_OPENAI_API_KEY e AZURE_OPENAI_ENDPOINT no .env da raiz "
            "(nomes legados AZURE_OPENAPI_KEY / AZURE_OPENAPI_BASE_URL também são aceitos)."
        )
    return api_key, base_url


def _build_llm() -> AzureChatOpenAI:
    api_key, base_url = _require_azure_env()
    return AzureChatOpenAI(
        azure_endpoint=base_url,
        api_key=api_key,
        azure_deployment=AZURE_DEPLOYMENT,
        api_version=AZURE_API_VERSION,
        temperature=0.2,
    )


def _format_deteccoes(deteccoes: list[dict[str, Any]]) -> str:
    if not deteccoes:
        return "(nenhuma detecção com posição disponível)"
    lines: list[str] = []
    for det in deteccoes:
        classe = det.get("class", "?")
        conf = det.get("confidence")
        conf_txt = f"{conf:.2f}" if isinstance(conf, (int, float)) else "n/a"
        lines.append(
            f"- {classe} conf={conf_txt} x={det.get('x')} y={det.get('y')}"
        )
    return "\n".join(lines)


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if fence:
        cleaned = fence.group(1).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(cleaned[start : end + 1])

    if not isinstance(data, dict):
        raise ValueError("Resposta do LLM não é um objeto JSON.")
    return data


def _normalizar_resultado(data: dict[str, Any]) -> dict[str, Any]:
    analise = data.get("analise_stride")
    if not isinstance(analise, dict):
        analise = {}

    vulnerabilidades = data.get("vulnerabilidades")
    if not isinstance(vulnerabilidades, list):
        vulnerabilidades = []

    vulns_norm: list[dict[str, Any]] = []
    for item in vulnerabilidades:
        if isinstance(item, dict):
            vulns_norm.append(item)
        elif isinstance(item, str) and item.strip():
            vulns_norm.append(
                {
                    "componente": "geral",
                    "risco": item.strip(),
                    "severidade": "media",
                    "categoria_stride": "I",
                    "contramedida": "",
                }
            )

    if "fluxo_inferido" in data and isinstance(data["fluxo_inferido"], str):
        analise = {
            **analise,
            "_fluxo_inferido": data["fluxo_inferido"],
        }

    return {
        "analise_stride": analise,
        "vulnerabilidades": vulns_norm,
    }


def _avisos_incerteza_deteccoes(deteccoes: list[dict[str, Any]]) -> list[str]:
    limiar = 0.55
    duvidosas: list[str] = []
    for det in deteccoes:
        if not isinstance(det, dict):
            continue
        conf = det.get("confidence")
        classe = det.get("class")
        if classe and isinstance(conf, (int, float)) and conf < limiar:
            duvidosas.append(f"{classe} ({conf:.2f})")
    if not duvidosas:
        return []
    unicas = list(dict.fromkeys(duvidosas))
    return [
        "Analista: a STRIDE abaixo herda incerteza da visão nas classes "
        f"{', '.join(unicas[:8])} — trate ameaças desses componentes como "
        "hipóteses, não como inventário confirmado."
    ]


def agente_analista(state: EstadoAmeacas) -> dict[str, Any]:
    """Aplica STRIDE via Azure OpenAI e atualiza o estado do grafo."""
    componentes = list(state.get("componentes_detectados") or [])
    deteccoes = list(state.get("deteccoes") or [])
    avisos: list[str] = []

    if not componentes and deteccoes:
        componentes = sorted(
            {
                str(det["class"]).strip()
                for det in deteccoes
                if isinstance(det, dict) and det.get("class")
            }
        )

    if not componentes:
        logger.warning("Nenhum componente para analisar; retornando análise vazia.")
        motivo = (
            "nenhum componente chegou da visão (falha, limiar de confiança ou "
            "diagrama fora do vocabulário)"
        )
        avisos.append(
            f"Analista: análise STRIDE não executada — {motivo}. "
            "O relatório não deve ser lido como ausência de ameaças."
        )
        return {
            "analise_stride": {},
            "vulnerabilidades": [],
            "avisos": avisos,
            "etapa_atual": "analista",
        }

    avisos.extend(_avisos_incerteza_deteccoes(deteccoes))

    prompt = USER_PROMPT_TEMPLATE.format(
        componentes=", ".join(componentes),
        deteccoes=_format_deteccoes(deteccoes),
    )

    try:
        llm = _build_llm()
        response = llm.invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        content = getattr(response, "content", None) or str(response)
        parsed = _extract_json(content)
        normalizado = _normalizar_resultado(parsed)
    except json.JSONDecodeError as exc:
        msg = (
            "Resposta do LLM não é JSON válido (parse falhou). "
            f"Análise STRIDE indisponível. Detalhe: {exc}"
        )
        logger.error("Falha na análise STRIDE: %s", msg)
        avisos.append(f"Analista: {msg}")
        return {
            "analise_stride": {"erro": msg},
            "vulnerabilidades": [],
            "avisos": avisos,
            "etapa_atual": "analista",
        }
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "DeploymentNotFound" in msg or "404" in msg:
            msg = (
                f"Deployment Azure OpenAI '{AZURE_DEPLOYMENT}' não encontrado. "
                "Crie um deployment no portal Azure (ou Azure AI Foundry) com esse nome, "
                f"ou defina AZURE_OPENAI_DEPLOYMENT no .env. Detalhe: {exc}"
            )
        logger.error("Falha na análise STRIDE: %s", msg)
        avisos.append(f"Analista: falha ao gerar STRIDE — {msg}")
        return {
            "analise_stride": {"erro": msg},
            "vulnerabilidades": [],
            "avisos": avisos,
            "etapa_atual": "analista",
        }

    fluxo = ""
    if isinstance(normalizado.get("analise_stride"), dict):
        fluxo = str(normalizado["analise_stride"].get("_fluxo_inferido") or "").strip()
    if fluxo:
        avisos.append(
            "Analista: o fluxo entre componentes é estimado a partir das posições "
            "(x, y) no diagrama — pode estar incompleto ou incorreto se a visão "
            "errou classes ou se o layout for ambíguo."
        )
    else:
        avisos.append(
            "Analista: não foi possível inferir um fluxo confiável entre "
            "componentes; a matriz STRIDE trata cada bloco de forma isolada."
        )

    if not normalizado["vulnerabilidades"]:
        avisos.append(
            "Analista: o LLM não retornou vulnerabilidades estruturadas "
            "(lista vazia após normalização)."
        )

    logger.info(
        "Analista: %d componentes, %d vulnerabilidades, %d avisos.",
        len(componentes),
        len(normalizado["vulnerabilidades"]),
        len(avisos),
    )
    return {
        **normalizado,
        "avisos": avisos,
        "etapa_atual": "analista",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    estado_demo: EstadoAmeacas = {
        "componentes_detectados": [
            "user",
            "api gateway",
            "identity provider",
            "serverless function",
            "web application",
        ],
        "deteccoes": [
            {"class": "user", "confidence": 0.68, "x": 64, "y": 416},
            {"class": "api gateway", "confidence": 0.55, "x": 420, "y": 280},
            {"class": "identity provider", "confidence": 0.50, "x": 220, "y": 120},
            {"class": "serverless function", "confidence": 0.48, "x": 620, "y": 260},
            {"class": "web application", "confidence": 0.45, "x": 800, "y": 320},
        ],
    }
    resultado = agente_analista(estado_demo)
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
