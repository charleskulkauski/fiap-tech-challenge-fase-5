"""Interface Streamlit — modelagem de ameaças a partir de diagramas."""

from __future__ import annotations

import base64
import html
import sys
import threading
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph import SAMPLE_IMAGE, SAMPLE_IMAGES, run_pipeline  # noqa: E402
from src.vision.label_aliases import display_label  # noqa: E402

UPLOAD_DIR = ROOT / "data" / "outputs"
UPLOAD_PATH = UPLOAD_DIR / "upload.jpg"
LOGO_PATH = ROOT / "data" / "fiap_logo.svg"

st.set_page_config(
    page_title="ThreatMap · FIAP",
    page_icon=str(LOGO_PATH) if LOGO_PATH.is_file() else "◇",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600;9..40,700&family=IBM+Plex+Mono:wght@400;500&display=swap');

      :root {
        --ink: #0a0a0a;
        --muted: #5c5c5c;
        --line: #e5e5e5;
        --panel: #f7f7f7;
        --accent: #e81b5c;
        --accent-dark: #b81448;
        --accent-soft: #fde8ef;
        --white: #ffffff;
      }

      html, body, [class*="css"] {
        font-family: "DM Sans", sans-serif;
        color: var(--ink);
      }

      .stApp {
        background:
          radial-gradient(ellipse 70% 45% at 0% -5%, #fde8ef 0%, transparent 55%),
          radial-gradient(ellipse 50% 35% at 100% 0%, #f0f0f0 0%, transparent 50%),
          var(--white);
      }

      .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 720px !important;
        margin-left: auto !important;
        margin-right: auto !important;
      }

      /* Header */
      .tm-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin: 0 0 1rem 0;
        width: 100%;
      }

      .tm-header img {
        width: 56px;
        height: 56px;
        flex-shrink: 0;
        border-radius: 50%;
        object-fit: cover;
        display: block;
      }

      .tm-header-text {
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 0.15rem;
        min-width: 0;
      }

      .tm-brand {
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--accent);
        margin: 0;
        line-height: 1.2;
      }

      .tm-title {
        font-size: clamp(1.75rem, 4vw, 2.35rem) !important;
        font-weight: 700 !important;
        line-height: 1.15 !important;
        letter-spacing: -0.03em;
        margin: 0 !important;
        color: var(--ink) !important;
        border-bottom: none !important;
        padding-bottom: 0 !important;
      }

      .tm-lead {
        color: var(--muted);
        font-size: 1.05rem;
        line-height: 1.55;
        margin: 0 0 0.35rem 0;
      }

      .tm-lead-block {
        margin: 0 0 1.5rem 0;
      }

      .tm-section {
        font-size: 1.35rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: var(--ink);
        margin: 1.5rem 0 0.65rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid var(--accent);
        display: inline-block;
        width: 100%;
      }

      .tm-section-sm {
        font-size: 1.15rem;
        font-weight: 600;
        letter-spacing: -0.015em;
        color: var(--ink);
        margin: 1.25rem 0 0.55rem 0;
      }

      [data-testid="stMarkdownContainer"] h1 {
        font-size: 1.28rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
        line-height: 1.3 !important;
        color: var(--ink) !important;
        margin-top: 1.1rem !important;
        margin-bottom: 0.45rem !important;
        border-bottom: 1px solid var(--line);
        padding-bottom: 0.35rem;
      }

      [data-testid="stMarkdownContainer"] h2 {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
        line-height: 1.35 !important;
        color: var(--ink) !important;
        margin-top: 1rem !important;
        margin-bottom: 0.35rem !important;
      }

      [data-testid="stMarkdownContainer"] h3 {
        font-size: 1rem !important;
        font-weight: 600 !important;
        line-height: 1.4 !important;
        color: var(--muted) !important;
        margin-top: 0.85rem !important;
        margin-bottom: 0.3rem !important;
      }

      .tm-chip {
        display: inline-block;
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.78rem;
        background: var(--accent-soft);
        color: var(--accent-dark);
        border: 1px solid #f5b8cc;
        border-radius: 4px;
        padding: 0.25rem 0.6rem;
        margin: 0.15rem 0.3rem 0.15rem 0;
      }

      .tm-meta {
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.78rem;
        color: var(--muted);
        border-top: 1px solid var(--line);
        padding-top: 0.75rem;
        margin-top: 0.35rem;
      }

      .tm-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.75rem;
      }

      div[data-testid="stFileUploader"] {
        background: var(--panel);
        border: 1px dashed #c9c9c9;
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.35rem;
      }

      div[data-testid="stCheckbox"] {
        margin: 0.5rem 0 0.85rem 0;
      }

      .stButton > button {
        background: var(--accent) !important;
        color: var(--white) !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em;
        padding: 0.5rem 1.25rem !important;
      }

      .stButton > button:hover {
        background: var(--accent-dark) !important;
        border-color: var(--accent-dark) !important;
      }

      .stProgress > div > div > div > div {
        background-color: var(--accent) !important;
      }

      hr {
        border-color: var(--line) !important;
        margin: 1.5rem 0 !important;
      }

      .stApp h1:not(.tm-title) {
        font-size: 1.28rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _salvar_upload(arquivo) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dados = arquivo.getvalue()
    UPLOAD_PATH.write_bytes(dados)
    return UPLOAD_PATH


def _chips(componentes: list[str]) -> str:
    if not componentes:
        return '<span class="tm-chip">nenhum componente</span>'
    return "".join(
        f'<span class="tm-chip">{html.escape(display_label(str(c)), quote=True)}</span>'
        for c in componentes
    )


def _rodar_com_progresso(caminho: Path):
    """Executa o pipeline com barra de progresso na UI."""
    holder: dict = {}

    def _worker() -> None:
        try:
            holder["resultado"] = run_pipeline(str(caminho.resolve()))
        except Exception as exc:  # noqa: BLE001
            holder["erro"] = exc

    progress = st.progress(0, text="Preparando análise…")
    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    etapas = [
        (12, "Detectando componentes (visão)…"),
        (38, "Aplicando análise STRIDE…"),
        (62, "Gerando relatório…"),
        (85, "Consolidando resultados…"),
    ]
    pct = 4
    etapa_idx = 0
    texto = "Iniciando pipeline…"

    while thread.is_alive():
        if etapa_idx < len(etapas) and pct >= etapas[etapa_idx][0]:
            texto = etapas[etapa_idx][1]
            etapa_idx += 1
        progress.progress(min(pct, 92), text=texto)
        time.sleep(0.35)
        pct = min(pct + 2, 92)

    thread.join()
    progress.progress(100, text="Análise concluída")
    time.sleep(0.25)
    progress.empty()

    if "erro" in holder:
        raise holder["erro"]
    return holder["resultado"]


if LOGO_PATH.is_file():
    _logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    _logo_img = (
        f'<img src="data:image/svg+xml;base64,{_logo_b64}" '
        'alt="FIAP" width="56" height="56" />'
    )
else:
    _logo_img = ""

st.markdown(
    f'<div class="tm-header">{_logo_img}'
    '<div class="tm-header-text">'
    '<p class="tm-brand">FIAP Hackathon</p>'
    '<h1 class="tm-title">Modelagem de Ameaças</h1>'
    "</div></div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="tm-lead-block">'
    '<p class="tm-lead">Ferramenta de modelagem de ameaças a partir de '
    "diagramas de arquitetura de software.</p>"
    '<p class="tm-lead">Envie uma arquitetura: o pipeline detecta componentes '
    "(visão), aplica STRIDE (analista) e gera o relatório final.</p>"
    "</div>",
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Diagrama (.jpeg, .jpg, .png)",
    type=["jpeg", "jpg", "png"],
    label_visibility="collapsed",
)

usar_exemplo = st.checkbox(
    "Usar diagrama de exemplo do PDF (avaliação FIAP)",
    value=uploaded is None,
)

exemplos_disponiveis = {
    rotulo: path for rotulo, path in SAMPLE_IMAGES.items() if path.is_file()
}
exemplo_escolhido = None
if usar_exemplo and exemplos_disponiveis:
    default_idx = 0
    rotulos = list(exemplos_disponiveis.keys())
    if SAMPLE_IMAGE.is_file():
        for i, path in enumerate(exemplos_disponiveis.values()):
            if path.resolve() == SAMPLE_IMAGE.resolve():
                default_idx = i
                break
    exemplo_escolhido = st.selectbox(
        "Diagrama de avaliação (exemplos utilizados no PDF do Hackathon)",
        options=rotulos,
        index=default_idx,
        help="Figuras 1 e 2 do PDF do Hackathon (IADT Fase 5).",
    )

# Preview do diagrama original (antes da anotação)
preview_bytes = None
preview_caption = None
if uploaded is not None:
    preview_bytes = uploaded.getvalue()
    preview_caption = f"Preview · {uploaded.name}"
elif usar_exemplo and exemplo_escolhido:
    _preview_path = exemplos_disponiveis[exemplo_escolhido]
    if _preview_path.is_file():
        preview_bytes = _preview_path.read_bytes()
        preview_caption = f"Preview · {exemplo_escolhido}"

if preview_bytes is not None:
    st.markdown(
        '<p class="tm-section-sm">Preview do diagrama</p>',
        unsafe_allow_html=True,
    )
    st.image(preview_bytes, use_container_width=True, caption=preview_caption)

rodar = st.button("Analisar diagrama", type="primary", use_container_width=True)

if rodar:
    if uploaded is not None:
        caminho = _salvar_upload(uploaded)
    elif usar_exemplo and exemplo_escolhido:
        caminho = exemplos_disponiveis[exemplo_escolhido]
    else:
        st.error("Envie uma imagem ou escolha um diagrama de exemplo do PDF.")
        st.stop()

    try:
        resultado = _rodar_com_progresso(caminho)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Falha no pipeline: {exc}")
        st.stop()

    st.session_state["resultado"] = resultado
    st.session_state["caminho_usado"] = str(caminho)

resultado = st.session_state.get("resultado")

if not resultado:
    st.markdown(
        '<p class="tm-meta">Aguardando análise · pipeline LangGraph linear</p>',
        unsafe_allow_html=True,
    )
    st.stop()

componentes = list(resultado.get("componentes_detectados") or [])
deteccoes = list(resultado.get("deteccoes") or [])
vulns = [v for v in (resultado.get("vulnerabilidades") or []) if isinstance(v, dict)]
analise = dict(resultado.get("analise_stride") or {})
avisos = [str(a) for a in (resultado.get("avisos") or []) if str(a).strip()]
relatorio = resultado.get("relatorio_final") or ""
imagem_anotada = resultado.get("imagem_anotada") or ""
etapa = resultado.get("etapa_atual") or "—"

st.markdown("---")
st.markdown('<p class="tm-section">Resultado</p>', unsafe_allow_html=True)
st.markdown(
    f'<p class="tm-meta">etapa: {etapa} · componentes: {len(componentes)} · '
    f"vulnerabilidades: {len(vulns)} · avisos: {len(avisos)}</p>",
    unsafe_allow_html=True,
)

if analise.get("erro"):
    st.error(f"Análise STRIDE indisponível: {analise['erro']}")
elif not componentes:
    st.warning(
        "Nenhum componente confirmado pela visão. O relatório não implica "
        "ausência de ameaças — veja os avisos abaixo."
    )

if avisos:
    st.markdown(
        '<p class="tm-section-sm">Avisos e limitações desta execução</p>',
        unsafe_allow_html=True,
    )
    for aviso in avisos:
        if aviso.lower().startswith("visão: falha") or "json" in aviso.lower():
            st.error(aviso)
        elif "confiança moderada" in aviso.lower() or "incerteza" in aviso.lower():
            st.warning(aviso)
        else:
            st.info(aviso)

st.markdown('<p class="tm-section-sm">Detecção anotada</p>', unsafe_allow_html=True)
if imagem_anotada and Path(imagem_anotada).is_file():
    st.image(imagem_anotada, use_container_width=True)
else:
    caminho_usado = st.session_state.get("caminho_usado")
    if caminho_usado and Path(caminho_usado).is_file():
        st.image(caminho_usado, use_container_width=True)
        st.caption("Sem bounding boxes — exibindo original.")
    else:
        st.info("Imagem anotada não disponível.")

st.markdown('<p class="tm-section-sm">Componentes</p>', unsafe_allow_html=True)
st.markdown(_chips(componentes), unsafe_allow_html=True)

if deteccoes:
    with st.expander(f"Detecções ({len(deteccoes)})", expanded=False):
        for det in deteccoes:
            conf = det.get("confidence")
            conf_txt = f"{conf:.2f}" if isinstance(conf, (int, float)) else "n/a"
            incerto = isinstance(conf, (int, float)) and conf < 0.55
            marca = " · incerto" if incerto else ""
            classe = display_label(det.get("class", "?"))
            st.write(
                f"`{classe}` · conf {conf_txt}{marca} · "
                f"({det.get('x')}, {det.get('y')})"
            )
else:
    st.caption("Sem detecções — visão vazia, limiar ou falha no YOLO local.")

with st.expander("Análise STRIDE", expanded=False):
    if analise.get("erro"):
        st.error(analise["erro"])
    elif not analise:
        st.caption("Sem análise STRIDE (visão sem componentes ou falha).")
    else:
        fluxo = analise.get("_fluxo_inferido")
        if fluxo:
            st.caption(f"Fluxo estimado (pode estar incorreto): {fluxo}")
        st.json(
            {k: v for k, v in analise.items() if not str(k).startswith("_")}
        )

with st.expander(f"Vulnerabilidades ({len(vulns)})", expanded=bool(vulns)):
    if not vulns:
        st.caption("Nenhuma vulnerabilidade retornada.")
    else:
        for item in vulns:
            st.markdown(
                f"**{item.get('severidade', '—').upper()}** · "
                f"`{display_label(item.get('componente', '—'))}` · "
                f"[{item.get('categoria_stride', '?')}]  \n"
                f"{item.get('risco', '')}  \n"
                f"_Contramedida:_ {item.get('contramedida', '—')}"
            )
            st.divider()

st.markdown('<p class="tm-section">Relatório final</p>', unsafe_allow_html=True)
st.markdown(relatorio)

st.download_button(
    label="Baixar relatório (.md)",
    data=relatorio.encode("utf-8"),
    file_name="relatorio-ameacas.md",
    mime="text/markdown",
)
