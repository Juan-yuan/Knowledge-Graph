"""Streamlit chat frontend for the FastAPI /process and /process_stream endpoints. Start with: bash streamlit/start_streamlit.sh"""
import html
import json
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import requests
import streamlit as st

# Must run before any project import so Knowledge-Graph modules resolve
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import Config
from common.logger_utils import get_logger, truncate

logger = get_logger("kg.streamlit", "streamlit.log")
conf = Config()

BRAND = conf.FASTAPI_TITLE
TAGLINE = conf.STREAMLIT_TAGLINE or ""
USER_ICON = "👤"
AI_ICON = "🌿"
EXAMPLE_QUESTIONS = conf.get_json("STREAMLIT_EXAMPLE_QUESTIONS", []) or []
SIDEBAR_HOWTO = conf.get_json("STREAMLIT_SIDEBAR_HOWTO", []) or []
DISCLAIMER = conf.STREAMLIT_DISCLAIMER or ""
INPUT_PLACEHOLDER = conf.STREAMLIT_INPUT_PLACEHOLDER or "Ask about a herb, formula, symptom, or disease…"
EMPTY_HINT = conf.STREAMLIT_EMPTY_HINT or "Try an example, or type your own question below."
SPINNER_TEXT = conf.STREAMLIT_SPINNER_TEXT or "Consulting the knowledge graph…"
STREAM_DEFAULT = bool(conf.STREAMLIT_STREAM_DEFAULT)
# Public API routes (must match fastapi/main.py); not secrets
PROCESS_PATH = "/process"
PROCESS_STREAM_PATH = "/process_stream"

st.set_page_config(
    page_title=BRAND,
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
  --ink: #1c2b24;
  --muted: #5c6b63;
  --accent: #2f6b55;
  --accent-soft: #e8f2ed;
  --user-soft: #eef2f0;
  --line: #d7e2db;
  --bg: #f6f8f6;
  --panel: #ffffff;
  --content-max: min(1080px, 92vw);
  --composer-h: 2.75rem;
}

html, body, [class*="css"] {
  font-family: "IBM Plex Sans", sans-serif;
  color: var(--ink);
}

.stApp {
  background:
    radial-gradient(1200px 500px at 10% -10%, #e7f1ea 0%, transparent 55%),
    radial-gradient(900px 400px at 100% 0%, #eef4f0 0%, transparent 50%),
    var(--bg);
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { display: none; }

div[data-testid="stMainBlockContainer"],
.block-container {
  max-width: var(--content-max) !important;
  padding-left: clamp(1rem, 3vw, 2rem) !important;
  padding-right: clamp(1rem, 3vw, 2rem) !important;
  padding-top: 1rem !important;
  padding-bottom: 1.5rem !important;
}

.brand-wrap {
  margin: 0 auto 0.85rem auto;
  text-align: center;
}
.brand-mark {
  font-size: 1.5rem;
  letter-spacing: 0.02em;
  margin-bottom: 0.25rem;
}
.brand-name {
  font-family: "Source Serif 4", Georgia, serif;
  font-weight: 600;
  font-size: clamp(1.55rem, 2.2vw, 1.9rem);
  line-height: 1.2;
  color: var(--ink);
  margin: 0;
}
.brand-tagline {
  margin: 0.45rem auto 0 auto;
  max-width: 40rem;
  color: var(--muted);
  font-size: 0.95rem;
  line-height: 1.5;
}
.status-row {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  margin-top: 0.7rem;
  padding: 0.28rem 0.7rem;
  border: 1px solid var(--line);
  border-radius: 999px;
  background: var(--panel);
  font-size: 0.78rem;
  color: var(--muted);
}
.status-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #b0b8b3;
}
.status-dot.ok { background: #2f6b55; }
.status-dot.bad { background: #a33b2d; }

.empty-hint {
  margin: 0.85rem auto 0.55rem auto;
  text-align: center;
  color: var(--muted);
  font-size: 0.9rem;
}
.disclaimer {
  margin: 0.9rem auto 0 auto;
  text-align: center;
  color: #7a8780;
  font-size: 0.75rem;
}

/* Chat bubbles: user left, AI right */
.msg-row {
  display: flex;
  align-items: flex-start;
  gap: 0.55rem;
  margin: 0.7rem 0;
  max-width: min(78%, 720px);
}
.msg-row.user {
  margin-right: auto;
  flex-direction: row;
}
.msg-row.ai {
  margin-left: auto;
  flex-direction: row-reverse;
}
.msg-icon {
  flex: 0 0 2rem;
  width: 2rem;
  height: 2rem;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1rem;
  background: var(--panel);
  border: 1px solid var(--line);
}
.msg-row.ai .msg-icon {
  background: var(--accent-soft);
  border-color: transparent;
}
.msg-bubble {
  padding: 0.75rem 0.95rem;
  border-radius: 14px;
  line-height: 1.55;
  font-size: 0.95rem;
  word-break: break-word;
}
.msg-row.user .msg-bubble {
  background: var(--user-soft);
  border: 1px solid var(--line);
  border-top-left-radius: 4px;
  color: var(--ink);
}
.msg-row.ai .msg-bubble {
  background: var(--panel);
  border: 1px solid var(--line);
  border-top-right-radius: 4px;
  color: var(--ink);
}

/* Composer: input + Send as one component (same control height) */
.ask-composer [data-testid="stForm"] {
  border: 1px solid var(--line);
  background: var(--panel);
  border-radius: 14px;
  padding: 0.4rem 0.45rem 0.4rem 0.65rem;
  margin-top: 0.85rem;
}
.ask-composer [data-testid="stForm"] [data-testid="stHorizontalBlock"] {
  align-items: center !important;
  gap: 0.5rem !important;
}
.ask-composer [data-testid="stForm"] [data-testid="column"] {
  display: flex !important;
  align-items: center !important;
}
.ask-composer [data-testid="stForm"] [data-testid="stTextInput"] {
  width: 100%;
}
.ask-composer [data-testid="stForm"] [data-testid="stTextInput"] label { display: none; }
.ask-composer [data-testid="stForm"] [data-testid="stTextInput"] > div,
.ask-composer [data-testid="stForm"] [data-testid="stTextInput"] > div > div {
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  height: var(--composer-h) !important;
  min-height: var(--composer-h) !important;
}
.ask-composer [data-testid="stForm"] input {
  font-family: "IBM Plex Sans", sans-serif !important;
  font-size: 0.95rem !important;
  height: var(--composer-h) !important;
  min-height: var(--composer-h) !important;
  max-height: var(--composer-h) !important;
  line-height: var(--composer-h) !important;
  padding: 0 0.35rem !important;
  margin: 0 !important;
  box-sizing: border-box !important;
}
.ask-composer [data-testid="stForm"] .stButton {
  width: 100%;
  height: var(--composer-h) !important;
  margin: 0 !important;
}
.ask-composer [data-testid="stForm"] .stButton > button {
  width: 100% !important;
  height: var(--composer-h) !important;
  min-height: var(--composer-h) !important;
  max-height: var(--composer-h) !important;
  margin: 0 !important;
  padding: 0 !important;
  background: var(--accent) !important;
  border: 1px solid var(--accent) !important;
  border-radius: 10px !important;
  color: #fff !important;
  font-weight: 600 !important;
  font-size: 0.92rem !important;
  line-height: 1 !important;
  white-space: nowrap !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  text-align: center !important;
  box-sizing: border-box !important;
}
.ask-composer [data-testid="stForm"] .stButton > button > div,
.ask-composer [data-testid="stForm"] .stButton > button p {
  margin: 0 !important;
  padding: 0 !important;
  line-height: 1 !important;
  width: 100% !important;
  height: 100% !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  text-align: center !important;
}
.ask-composer [data-testid="stForm"] .stButton > button:hover {
  background: #275a48 !important;
  border-color: #275a48 !important;
  color: #fff !important;
}

section[data-testid="stSidebar"] {
  background: #f3f6f4;
  border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
  font-family: "Source Serif 4", Georgia, serif;
  font-weight: 600;
  color: var(--ink);
}

.stButton > button {
  border: 1px solid var(--line);
  background: var(--panel);
  color: var(--ink);
  border-radius: 10px;
  font-weight: 500;
}
.stButton > button:hover {
  border-color: var(--accent);
  color: var(--accent);
}

/* Example prompts: 3-column row only (composer/sidebar are 2 columns).
   Support both data-testid="column" and "stColumn" across Streamlit versions. */
div[data-testid="stHorizontalBlock"]:has(> :nth-child(3)):not(:has(> :nth-child(4))) {
  align-items: stretch !important;
  gap: 0.65rem !important;
}
div[data-testid="stHorizontalBlock"]:has(> :nth-child(3)):not(:has(> :nth-child(4)))
  > [data-testid="column"],
div[data-testid="stHorizontalBlock"]:has(> :nth-child(3)):not(:has(> :nth-child(4)))
  > [data-testid="stColumn"] {
  display: flex !important;
}
div[data-testid="stHorizontalBlock"]:has(> :nth-child(3)):not(:has(> :nth-child(4))) .stButton {
  width: 100% !important;
  height: 100% !important;
}
div[data-testid="stHorizontalBlock"]:has(> :nth-child(3)):not(:has(> :nth-child(4))) .stButton > button {
  width: 100% !important;
  height: 6.25rem !important;
  min-height: 6.25rem !important;
  max-height: 6.25rem !important;
  display: flex !important;
  align-items: center !important;
  justify-content: flex-start !important;
  text-align: left !important;
  white-space: normal !important;
  overflow: hidden !important;
  padding: 0.75rem 0.9rem !important;
  background: #e8f2ed !important;
  border: 1px solid transparent !important;
  border-radius: 10px !important;
  color: #1c2b24 !important;
  font-size: 0.86rem !important;
  font-weight: 500 !important;
  line-height: 1.35 !important;
  box-shadow: none !important;
}
div[data-testid="stHorizontalBlock"]:has(> :nth-child(3)):not(:has(> :nth-child(4))) .stButton > button:hover {
  background: #dceae3 !important;
  border-color: #2f6b55 !important;
  color: #1c2b24 !important;
}
div[data-testid="stHorizontalBlock"]:has(> :nth-child(3)):not(:has(> :nth-child(4))) .stButton > button > div,
div[data-testid="stHorizontalBlock"]:has(> :nth-child(3)):not(:has(> :nth-child(4))) .stButton > button p {
  text-align: left !important;
  white-space: normal !important;
  width: 100% !important;
}
</style>
""",
    unsafe_allow_html=True,
)

if "user_id" not in st.session_state:
    st.session_state.user_id = f"web-{uuid.uuid4().hex[:8]}"
    st.session_state.messages = []
    st.session_state.api_ok = None
    logger.info("New session user_id=%s, API_BASE_URL=%s", st.session_state.user_id, conf.API_BASE_URL)

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

is_home = not st.session_state.messages and not st.session_state.pending_question

if is_home:
    # Only on the empty home screen: center brand / examples / input in the viewport
    st.markdown(
        """
<style>
[data-testid="stAppViewContainer"] > .main,
[data-testid="stMain"] {
  display: flex !important;
  flex-direction: column !important;
  justify-content: center !important;
  min-height: 100vh !important;
}
[data-testid="stMain"] > div {
  width: 100%;
}
div[data-testid="stMainBlockContainer"],
.block-container {
  min-height: calc(100vh - 1rem) !important;
  display: flex !important;
  flex-direction: column !important;
  justify-content: center !important;
  /* Optical shift: slightly above true center */
  padding-top: 0 !important;
  padding-bottom: 10vh !important;
  transform: translateY(-3.5vh);
}
</style>
""",
        unsafe_allow_html=True,
    )


def normalize_api_base_url(url: str) -> tuple[str, bool]:
    """Allow only http(s) base URLs without embedded credentials. Returns (url, ok)."""
    default = (conf.API_BASE_URL or "http://localhost:8000").rstrip("/")
    cleaned = (url or "").strip().rstrip("/")
    if not cleaned:
        return default, False
    parsed = urlparse(cleaned)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return default, False
    # Avoid leaking user:pass@host via logs / sidebar if pasted by mistake
    if parsed.username or parsed.password:
        return default, False
    return cleaned, True


def check_health(base_url: str) -> tuple[bool, str]:
    try:
        response = requests.get(f"{base_url}/", timeout=conf.API_HEALTH_TIMEOUT)
        logger.info("Health check %s -> %d %s", base_url, response.status_code, truncate(response.text))
        response.raise_for_status()
        return True, response.json().get("message", "ok")
    except requests.RequestException as e:
        logger.warning("Health check %s failed: %s", base_url, e)
        return False, str(e)


def ask(base_url: str, question: str, user_id: str) -> str:
    """Call POST /process and return the answer text (or a readable error message)."""
    request_id = uuid.uuid4().hex[:8]
    logger.info("[%s] POST %s%s user_id=%s input=%r", request_id, base_url, PROCESS_PATH, user_id, question)
    start = time.perf_counter()
    try:
        response = requests.post(
            f"{base_url}{PROCESS_PATH}",
            json={"input": question, "user_id": user_id},
            headers={"X-Request-ID": request_id},
            timeout=conf.API_PROCESS_TIMEOUT,
        )
    except requests.Timeout:
        logger.error("[%s] %s timed out after %ds", request_id, PROCESS_PATH, conf.API_PROCESS_TIMEOUT)
        return f"Request timed out after {conf.API_PROCESS_TIMEOUT}s. Please try again."
    except requests.ConnectionError as e:
        logger.error("[%s] Cannot connect to %s: %s", request_id, base_url, e)
        return f"Cannot connect to the API at {base_url}. Is the FastAPI server running?"

    elapsed = time.perf_counter() - start
    logger.info("[%s] %s -> %d in %.2fs", request_id, PROCESS_PATH, response.status_code, elapsed)
    try:
        data = response.json()
    except ValueError:
        logger.error("[%s] Non-JSON response: %s", request_id, truncate(response.text))
        return f"Unexpected response from the API (HTTP {response.status_code})."

    if response.status_code != 200:
        logger.error("[%s] %s error body: %s", request_id, PROCESS_PATH, truncate(data))
        return f"{data.get('output') or data.get('detail') or 'API error'} (HTTP {response.status_code})"

    logger.debug("[%s] Answer: %s", request_id, truncate(data.get("output", ""), 2000))
    return data.get("output", "") or "No answer was returned."


def render_step(step: dict) -> None:
    """One finished graph node: its name plus the state fields it changed."""
    # Backticks reduce markdown injection from untrusted node names
    node = str(step.get("node") or "?").replace("`", "'")
    st.markdown(f"✅ `{node}`")
    if step.get("data"):
        st.json(step["data"], expanded=False)


def render_thinking(steps: list, elapsed: float) -> None:
    """Collapsed thinking process shown above a past streamed answer."""
    with st.expander(f"Thinking process · {len(steps)} steps · {elapsed:.1f}s", expanded=False):
        for step in steps:
            render_step(step)


def ask_stream(base_url: str, question: str, user_id: str) -> tuple[str, list, float]:
    """Call POST /process_stream, showing each node live in an expanded status box that
    collapses once the final answer arrives. Returns (answer, thinking steps, elapsed seconds)."""
    request_id = uuid.uuid4().hex[:8]
    logger.info(
        "[%s] POST %s%s user_id=%s input=%r",
        request_id,
        base_url,
        PROCESS_STREAM_PATH,
        user_id,
        question,
    )
    start = time.perf_counter()
    steps: list = []
    answer, state = None, "complete"
    status = st.status("Thinking…", expanded=True)
    slots: dict = {}
    try:
        with requests.post(
            f"{base_url}{PROCESS_STREAM_PATH}",
            json={"input": question, "user_id": user_id},
            headers={"X-Request-ID": request_id},
            stream=True,
            timeout=conf.API_PROCESS_TIMEOUT,
        ) as response:
            if response.status_code != 200:
                logger.error(
                    "[%s] %s -> %d: %s",
                    request_id,
                    PROCESS_STREAM_PATH,
                    response.status_code,
                    truncate(response.text),
                )
                answer, state = f"API error (HTTP {response.status_code}).", "error"
            else:
                pending: dict = {}
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        message = json.loads(line)
                    except json.JSONDecodeError:
                        logger.error("[%s] Malformed stream line: %s", request_id, truncate(line))
                        answer, state = "Unexpected response from the API.", "error"
                        break
                    if not isinstance(message, dict):
                        logger.error("[%s] Non-object stream line: %s", request_id, truncate(line))
                        answer, state = "Unexpected response from the API.", "error"
                        break
                    msg_type = message.get("type")
                    if msg_type != "thinking":
                        answer = message.get("output") or "No answer was returned."
                        if msg_type == "error":
                            state = "error"
                        break
                    node = message.get("node")
                    node_status = message.get("status")
                    if not node or not node_status:
                        logger.error("[%s] Thinking event missing node/status: %s", request_id, truncate(message))
                        continue
                    node = str(node)
                    if node_status == "start":
                        safe_node = node.replace("`", "'")
                        status.update(label=f"Thinking… running {safe_node}")
                        slots[node] = status.empty()
                        slots[node].markdown(f"⏳ `{safe_node}` …")
                    elif node_status == "update":
                        pending[node] = message.get("data") or {}
                    elif node_status == "end":
                        step = {"node": node, "data": pending.pop(node, {})}
                        steps.append(step)
                        with slots.pop(node, status.empty()).container():
                            render_step(step)
    except requests.Timeout:
        logger.error("[%s] %s timed out after %ds", request_id, PROCESS_STREAM_PATH, conf.API_PROCESS_TIMEOUT)
        answer, state = f"Request timed out after {conf.API_PROCESS_TIMEOUT}s. Please try again.", "error"
    except requests.ConnectionError as e:
        logger.error("[%s] Cannot connect to %s: %s", request_id, base_url, e)
        answer, state = f"Cannot connect to the API at {base_url}. Is the FastAPI server running?", "error"
    except (TypeError, KeyError, ValueError):
        logger.exception("[%s] Malformed stream payload", request_id)
        answer, state = "Unexpected response from the API.", "error"

    if answer is None:
        logger.error("[%s] Stream ended without a final message", request_id)
        answer, state = "The answer stream ended unexpectedly. Please try again.", "error"
    elapsed = time.perf_counter() - start
    logger.info("[%s] %s done in %.2fs, %d steps", request_id, PROCESS_STREAM_PATH, elapsed, len(steps))
    logger.debug("[%s] Answer: %s", request_id, truncate(answer, 2000))
    status.update(
        label=f"Thinking process · {len(steps)} steps · {elapsed:.1f}s",
        state=state,
        expanded=False,
    )
    return answer, steps, elapsed


def submit_question(question: str) -> None:
    st.session_state.pending_question = question.strip()


def render_message(role: str, content: str) -> None:
    """Render a chat bubble: user on the left, AI on the right, each with an icon."""
    safe = html.escape(content).replace("\n", "<br>")
    if role == "user":
        st.markdown(
            f'<div class="msg-row user"><div class="msg-icon">{USER_ICON}</div>'
            f'<div class="msg-bubble">{safe}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="msg-row ai"><div class="msg-icon">{AI_ICON}</div>'
            f'<div class="msg-bubble">{safe}</div></div>',
            unsafe_allow_html=True,
        )


def render_ask_composer(form_key: str) -> None:
    """Single composer: text field + Send button on the right."""
    st.markdown('<div class="ask-composer">', unsafe_allow_html=True)
    with st.form(form_key, clear_on_submit=True):
        input_col, send_col = st.columns([8, 1.2])
        with input_col:
            question = st.text_input(
                "Question",
                placeholder=INPUT_PLACEHOLDER,
                label_visibility="collapsed",
            )
        with send_col:
            sent = st.form_submit_button("Send", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    if sent and question.strip():
        submit_question(question)
        st.rerun()


with st.sidebar:
    st.markdown("### Connection")
    raw_api_base_url = st.text_input("API base URL", value=conf.API_BASE_URL)
    api_base_url, api_url_ok = normalize_api_base_url(raw_api_base_url)
    if not api_url_ok and (raw_api_base_url or "").strip():
        st.warning("Invalid API base URL (use http(s)://host[:port] without credentials). Using the configured default.")
    status_placeholder = st.empty()

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Check status", use_container_width=True):
            ok, message = check_health(api_base_url)
            st.session_state.api_ok = ok
            status_placeholder.success(message) if ok else status_placeholder.error(message)
    with col_b:
        if st.button("New chat", use_container_width=True):
            logger.info(
                "user_id=%s cleared %d messages",
                st.session_state.user_id,
                len(st.session_state.messages),
            )
            st.session_state.messages = []
            st.session_state.pending_question = None
            st.rerun()

    stream_enabled = st.toggle(
        "Stream thinking",
        value=STREAM_DEFAULT,
        help=f"Use {PROCESS_STREAM_PATH} to show each reasoning step live; off uses {PROCESS_PATH}.",
    )

    st.caption(f"Session `{st.session_state.user_id}`")
    if SIDEBAR_HOWTO:
        st.divider()
        steps = "\n".join(f"{i}. {step}" for i, step in enumerate(SIDEBAR_HOWTO, start=1))
        st.markdown(f"**How it works**\n\n{steps}")

if st.session_state.api_ok is None:
    st.session_state.api_ok, _ = check_health(api_base_url)

api_label = "API online" if st.session_state.api_ok else "API offline"
api_class = "ok" if st.session_state.api_ok else "bad"

st.markdown(
    f"""
<div class="brand-wrap">
  <div class="brand-mark">🌿</div>
  <h1 class="brand-name">{html.escape(BRAND)}</h1>
  <p class="brand-tagline">{html.escape(TAGLINE)}</p>
  <div class="status-row"><span class="status-dot {api_class}"></span>{html.escape(api_label)}</div>
</div>
""",
    unsafe_allow_html=True,
)

if is_home:
    st.markdown(
        f'<p class="empty-hint">{html.escape(EMPTY_HINT)}</p>',
        unsafe_allow_html=True,
    )
    cols = st.columns(3, gap="small")
    for idx, (col, example) in enumerate(zip(cols, EXAMPLE_QUESTIONS)):
        with col:
            if st.button(example, key=f"ex_{idx}", use_container_width=True):
                submit_question(example)
                st.rerun()
    render_ask_composer("home_ask")
    if DISCLAIMER:
        st.markdown(
            f'<p class="disclaimer">{html.escape(DISCLAIMER)}</p>',
            unsafe_allow_html=True,
        )
else:
    for message in st.session_state.messages:
        if message.get("thinking"):
            render_thinking(message["thinking"], float(message.get("elapsed") or 0.0))
        render_message(message["role"], message["content"])
    render_ask_composer("chat_ask")

if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None
    st.session_state.messages.append({"role": "user", "content": question})
    render_message("user", question)

    if stream_enabled:
        answer, steps, elapsed = ask_stream(api_base_url, question, st.session_state.user_id)
        reply = {"role": "assistant", "content": answer, "thinking": steps, "elapsed": elapsed}
    else:
        with st.spinner(SPINNER_TEXT):
            answer = ask(api_base_url, question, st.session_state.user_id)
        reply = {"role": "assistant", "content": answer}
    render_message("assistant", answer)
    st.session_state.messages.append(reply)
