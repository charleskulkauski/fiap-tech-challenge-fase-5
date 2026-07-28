"""Nó LangGraph: consolida o estado em relatório markdown final."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.state.estado_ameaca import EstadoAmeacas  # noqa: E402
from src.vision.label_aliases import display_label  # noqa: E402

logger = logging.getLogger(__name__)

OUTPUT_REPORT = _ROOT / "data" / "outputs" / "relatorio.md"

STRIDE_LABELS = {
    "S": "Spoofing",
    "T": "Tampering",
    "R": "Repudiation",
    "I": "Information Disclosure",
    "D": "Denial of Service",
    "E": "Elevation of Privilege",
}

SEVERIDADE_ORDEM = {"alta": 0, "media": 1, "média": 1, "baixa": 2}


def _esc(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def _resumo_executivo(
    componentes: list[str],
    vulnerabilidades: list[dict[str, Any]],
    fluxo: str,
    avisos: list[str],
    analise: dict[str, Any],
) -> str:
    n_comp = len(componentes)
    n_vuln = len(vulnerabilidades)
    altas = sum(
        1
        for v in vulnerabilidades
        if str(v.get("severidade", "")).lower() in {"alta", "high"}
    )
    comps = (
        ", ".join(display_label(c) for c in componentes)
        if componentes
        else "nenhum componente identificado"
    )
    fluxo_txt = fluxo or "fluxo entre componentes não inferido"

    base = (
        f"Este relatório apresenta a modelagem de ameaças (STRIDE) de um diagrama "
        f"de arquitetura com **{n_comp}** componente(s) identificado(s) ({comps}). "
        f"Foram catalogadas **{n_vuln}** vulnerabilidade(s), das quais **{altas}** "
        f"com severidade alta. Fluxo observado: {fluxo_txt}."
    )

    qualificadores: list[str] = []
    if analise.get("erro"):
        qualificadores.append(
            "**Atenção:** a análise STRIDE falhou; o conteúdo abaixo é parcial "
            "ou vazio e não substitui revisão humana."
        )
    elif not componentes:
        qualificadores.append(
            "**Atenção:** nenhum componente foi confirmado pela visão — "
            "ausência de vulnerabilidades listadas **não** significa arquitetura segura."
        )
    elif avisos:
        qualificadores.append(
            f"**Atenção:** há **{len(avisos)}** aviso(s) de limitação/incerteza "
            "(ver seção 6)."
        )

    if not qualificadores:
        return base
    return base + "\n\n" + "\n\n".join(qualificadores)


def _secao_limitacoes(avisos: list[str], analise: dict[str, Any]) -> str:
    genericas = [
        "A detecção de componentes depende do YOLOv8 treinado no Colab "
        "(dataset de diagramas e qualidade das anotações).",
        "A análise STRIDE é assistida por LLM e pode generalizar riscos; "
        "não substitui revisão humana nem pentest.",
        "Contramedidas são orientações genéricas (OWASP/cloud), sem CVEs "
        "específicos inventados.",
    ]
    linhas = [f"- {item}" for item in genericas]

    if analise.get("erro"):
        linhas.append(f"- Erro do analista: {_esc(analise['erro'])}")

    if avisos:
        linhas.append("")
        linhas.append("### Avisos observados nesta execução")
        linhas.append("")
        for aviso in avisos:
            linhas.append(f"- {_esc(aviso)}")
    else:
        linhas.append(
            "- Nenhum aviso operacional adicional foi registrado nesta execução."
        )

    return "\n".join(linhas)


def _secao_componentes(componentes: list[str]) -> str:
    if not componentes:
        return (
            "_Nenhum componente detectado pelo agente de visão._\n\n"
            "_Isso pode significar falha na API, limiar de confiança, diagrama "
            "ambíguo/incompleto ou classes fora do vocabulário — não ausência de risco._"
        )
    linhas = [f"- `{display_label(c)}`" for c in componentes]
    return "\n".join(linhas)


def _celula_ameaca(entrada: Any) -> str:
    if isinstance(entrada, dict):
        ameaca = entrada.get("ameaca") or entrada.get("descricao") or ""
        return _esc(ameaca) if ameaca else "—"
    if isinstance(entrada, str) and entrada.strip():
        return _esc(entrada)
    return "—"


def _secao_stride(analise: dict[str, Any]) -> str:
    componentes = {
        k: v
        for k, v in analise.items()
        if not str(k).startswith("_") and isinstance(v, dict)
    }
    if not componentes:
        if analise.get("erro"):
            return f"_Análise STRIDE indisponível: {_esc(analise['erro'])}_"
        return "_Análise STRIDE vazia._"

    partes: list[str] = []
    for nome, categorias in componentes.items():
        partes.append(f"### {display_label(nome)}")
        partes.append("")
        partes.append("| Categoria | Ameaça | Contramedida |")
        partes.append("|---|---|---|")
        for letra, rotulo in STRIDE_LABELS.items():
            entrada = categorias.get(letra) or categorias.get(rotulo) or {}
            ameaca = _celula_ameaca(entrada)
            contramedida = "—"
            if isinstance(entrada, dict):
                contramedida = _esc(entrada.get("contramedida") or "—")
            partes.append(f"| **{letra}** ({rotulo}) | {ameaca} | {contramedida} |")
        partes.append("")
    return "\n".join(partes).rstrip()


def _ordenar_vulns(vulnerabilidades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def chave(item: dict[str, Any]) -> tuple[int, str]:
        sev = str(item.get("severidade", "")).lower()
        return (SEVERIDADE_ORDEM.get(sev, 9), display_label(str(item.get("componente", ""))))

    return sorted(vulnerabilidades, key=chave)


def _secao_vulnerabilidades(vulnerabilidades: list[dict[str, Any]]) -> str:
    if not vulnerabilidades:
        return "_Nenhuma vulnerabilidade listada._"

    linhas = [
        "| Severidade | Componente | Risco | STRIDE | Contramedida |",
        "|---|---|---|---|---|",
    ]
    for item in _ordenar_vulns(vulnerabilidades):
        linhas.append(
            "| {sev} | {comp} | {risco} | {cat} | {ctrl} |".format(
                sev=_esc(item.get("severidade", "—")),
                comp=_esc(display_label(item.get("componente", "—"))),
                risco=_esc(item.get("risco", "—")),
                cat=_esc(item.get("categoria_stride", "—")),
                ctrl=_esc(item.get("contramedida", "—")),
            )
        )
    return "\n".join(linhas)


def _secao_contramedidas(vulnerabilidades: list[dict[str, Any]], analise: dict[str, Any]) -> str:
    vistas: set[str] = set()
    bullets: list[str] = []

    for item in _ordenar_vulns(vulnerabilidades):
        ctrl = str(item.get("contramedida") or "").strip()
        if not ctrl or ctrl in vistas:
            continue
        vistas.add(ctrl)
        comp = display_label(item.get("componente", "geral"))
        bullets.append(f"- **{comp}**: {ctrl}")

    for nome, categorias in analise.items():
        if str(nome).startswith("_") or not isinstance(categorias, dict):
            continue
        for letra, entrada in categorias.items():
            if not isinstance(entrada, dict):
                continue
            ctrl = str(entrada.get("contramedida") or "").strip()
            if not ctrl or ctrl in vistas:
                continue
            vistas.add(ctrl)
            bullets.append(f"- **{display_label(nome)}** ({letra}): {ctrl}")

    if not bullets:
        return "_Nenhuma contramedida explícita registrada._"
    return "\n".join(bullets)


def gerar_relatorio_markdown(state: EstadoAmeacas) -> str:
    """Monta o markdown do relatório a partir do estado."""
    componentes = list(state.get("componentes_detectados") or [])
    analise = dict(state.get("analise_stride") or {})
    vulnerabilidades = [
        v for v in (state.get("vulnerabilidades") or []) if isinstance(v, dict)
    ]
    avisos = [str(a) for a in (state.get("avisos") or []) if str(a).strip()]
    caminho = state.get("caminho_imagem") or "—"
    imagem_anotada = state.get("imagem_anotada") or ""
    fluxo = str(analise.get("_fluxo_inferido") or "").strip()
    agora = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    secoes = [
        "# Relatório de Modelagem de Ameaças (STRIDE)",
        "",
        f"_Gerado em {agora}_",
        "",
        f"**Diagrama analisado:** `{caminho}`",
    ]
    if imagem_anotada:
        secoes.append(f"**Imagem anotada:** `{imagem_anotada}`")
    if avisos:
        secoes.append(f"**Avisos nesta execução:** {len(avisos)} (detalhes na seção 6)")
    secoes.extend(
        [
            "",
            "## 1. Resumo executivo",
            "",
            _resumo_executivo(componentes, vulnerabilidades, fluxo, avisos, analise),
            "",
            "## 2. Componentes identificados",
            "",
            _secao_componentes(componentes),
            "",
            "## 3. Matriz STRIDE",
            "",
            _secao_stride(analise),
            "",
            "## 4. Vulnerabilidades priorizadas",
            "",
            _secao_vulnerabilidades(vulnerabilidades),
            "",
            "## 5. Contramedidas recomendadas",
            "",
            _secao_contramedidas(vulnerabilidades, analise),
            "",
            "## 6. Limitações e avisos desta execução",
            "",
            _secao_limitacoes(avisos, analise),
            "",
        ]
    )
    return "\n".join(secoes)


def agente_relatorio(state: EstadoAmeacas) -> dict[str, Any]:
    """Consolida o estado em relatorio_final (markdown)."""
    relatorio = gerar_relatorio_markdown(state)

    try:
        OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_REPORT.write_text(relatorio, encoding="utf-8")
        logger.info("Relatório salvo em %s", OUTPUT_REPORT)
    except OSError as exc:
        logger.warning("Não foi possível gravar %s: %s", OUTPUT_REPORT, exc)

    return {
        "relatorio_final": relatorio,
        "etapa_atual": "relatorio",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    estado_demo: EstadoAmeacas = {
        "caminho_imagem": str(
            _ROOT / "data" / "material-fiap" / "diagrama-arquitetura.jpeg"
        ),
        "imagem_anotada": str(
            _ROOT / "data" / "outputs" / "diagrama-arquitetura-annotated.jpg"
        ),
        "componentes_detectados": [
            "user",
            "api gateway",
            "identity provider",
            "serverless function",
        ],
        "analise_stride": {
            "_fluxo_inferido": "user -> identity provider -> api gateway -> backends",
            "api gateway": {
                "S": {
                    "ameaca": "Tokens falsificados nas chamadas HTTP",
                    "contramedida": "Validar JWT no gateway e mTLS entre serviços",
                },
                "T": {
                    "ameaca": "Alteração de payload em trânsito",
                    "contramedida": "TLS 1.2+ e assinatura de requisições",
                },
                "R": {
                    "ameaca": "Falta de trilha de auditoria de chamadas",
                    "contramedida": "Logs estruturados com correlation id",
                },
                "I": {
                    "ameaca": "Exposição de dados sensíveis em respostas de erro",
                    "contramedida": "Mascarar erros e filtrar PII",
                },
                "D": {
                    "ameaca": "Saturação do gateway",
                    "contramedida": "Rate limiting e WAF",
                },
                "E": {
                    "ameaca": "Políticas permissivas de API",
                    "contramedida": "Least privilege e revisão de products/APIs",
                },
            },
        },
        "vulnerabilidades": [
            {
                "componente": "api gateway",
                "risco": "API sem rate limiting",
                "severidade": "alta",
                "categoria_stride": "D",
                "contramedida": "Habilitar rate limiting e quotas por assinatura",
            },
            {
                "componente": "identity provider",
                "risco": "Ausência de MFA para clientes privilegiados",
                "severidade": "media",
                "categoria_stride": "S",
                "contramedida": "Exigir MFA e Conditional Access",
            },
        ],
        "avisos": [
            "Visão: 1 detecção(ões) com confiança moderada (<0.55) — a classe "
            "pode estar errada: serverless function (0.48).",
            "Analista: o fluxo entre componentes é estimado a partir das posições "
            "(x, y) no diagrama — pode estar incompleto ou incorreto se a visão "
            "errou classes ou se o layout for ambíguo.",
        ],
    }

    resultado = agente_relatorio(estado_demo)
    print(resultado["relatorio_final"])
    print("---")
    print("salvo em:", OUTPUT_REPORT)
