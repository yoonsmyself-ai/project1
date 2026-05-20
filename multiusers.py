"""멀티유저/멀티세션 RAG 챗봇 — DB user 테이블 로그인, Supabase 세션·벡터 저장."""

from __future__ import annotations

import logging
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import bcrypt
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import Client, create_client

# ---------------------------------------------------------------------------
# Paths & environment
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
LOGO_PATH = REPO_ROOT / "logo.png"
LOG_DIR = REPO_ROOT / "logs"


def _writable_log_dir() -> Path:
    """Local dev: repo logs/. Streamlit Cloud: fall back to system temp (read-only mount)."""
    for candidate in (LOG_DIR, Path(tempfile.gettempdir()) / "ai-education-logs"):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError:
            continue
    return Path(tempfile.gettempdir())


load_dotenv(dotenv_path=ENV_PATH)

MODEL_NAME = "gpt-4o-mini"
EMBEDDING_DIM = 1536
VECTOR_BATCH_SIZE = 10
USER_TABLE = "user"

ANSWER_STYLE_SYSTEM = """당신은 친절하고 공손한 AI 어시스턴트입니다.

답변 규칙:
- 반드시 마크다운 헤딩(# ## ###)으로 구조화하세요. 주요 주제는 #, 세부는 ##, 구체 설명은 ###.
- 서술형으로 완전한 문장을 사용하고 존댓말로 작성하세요.
- 구분선(---, ===, ___)은 사용하지 마세요.
- 취소선(~~텍스트~~)은 사용하지 마세요.
- 참조 표시, 각주, 출처 문구, URL 인용 문장은 넣지 마세요.
"""


def _setup_logging() -> logging.Logger:
    log_dir = _writable_log_dir()
    log_path = log_dir / f"multiusers_{datetime.now().strftime('%Y%m%d')}.log"
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        handlers.insert(0, logging.FileHandler(log_path, encoding="utf-8"))
    except OSError:
        pass
    for handler in handlers:
        handler.setLevel(logging.WARNING)
        handler.setFormatter(fmt)
        root.addHandler(handler)
    for name in ("httpx", "httpcore", "urllib3", "openai", "langchain", "langchain_openai"):
        logging.getLogger(name).setLevel(logging.WARNING)
    return logging.getLogger("multiusers")


logger = _setup_logging()


def remove_separators(text: str) -> str:
    out = re.sub(r"~~([^~]*)~~", r"\1", text)
    out = re.sub(r"(?m)^\s*-{3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*={3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*_{3,}\s*$", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _get_secret(key: str) -> str:
    """Streamlit Cloud secrets 우선, 없으면 .env / os.getenv."""
    try:
        value = st.secrets.get(key)
        if value:
            return str(value).strip()
    except Exception:
        pass
    return os.getenv(key, "").strip()


def _env_keys() -> dict[str, str]:
    return {
        "openai": _get_secret("OPENAI_API_KEY"),
        "supabase_url": _get_secret("SUPABASE_URL"),
        "supabase_anon": _get_secret("SUPABASE_ANON_KEY"),
    }


def _missing_keys(keys: dict[str, str]) -> list[str]:
    missing: list[str] = []
    if not keys["openai"]:
        missing.append("OPENAI_API_KEY")
    if not keys["supabase_url"]:
        missing.append("SUPABASE_URL")
    if not keys["supabase_anon"]:
        missing.append("SUPABASE_ANON_KEY")
    return missing


def get_supabase_client(keys: dict[str, str]) -> Client | None:
    if _missing_keys(keys):
        return None
    return create_client(keys["supabase_url"], keys["supabase_anon"])


def get_llm(openai_key: str, temperature: float = 0.7) -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=temperature,
        api_key=openai_key,
    )


def get_embeddings(openai_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(api_key=openai_key)


# ---------------------------------------------------------------------------
# Password & auth (public."user" 테이블)
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def register_user(sb: Client, login_id: str, password: str) -> tuple[bool, str]:
    login_id = login_id.strip()
    if not login_id or not password:
        return False, "아이디와 비밀번호를 입력하세요."
    if len(password) < 4:
        return False, "비밀번호는 4자 이상이어야 합니다."
    existing = (
        sb.table(USER_TABLE)
        .select("id")
        .eq("login_id", login_id)
        .limit(1)
        .execute()
    )
    if existing.data:
        return False, "이미 사용 중인 아이디입니다."
    try:
        sb.table(USER_TABLE).insert(
            {
                "login_id": login_id,
                "password_hash": hash_password(password),
            }
        ).execute()
        return True, "회원가입이 완료되었습니다. 로그인해 주세요."
    except Exception as exc:  # noqa: BLE001
        logger.warning("회원가입 실패: %s", exc)
        return False, f"회원가입 중 오류가 발생했습니다: {exc}"


def login_user(sb: Client, login_id: str, password: str) -> tuple[bool, str, str | None]:
    login_id = login_id.strip()
    if not login_id or not password:
        return False, "아이디와 비밀번호를 입력하세요.", None
    resp = (
        sb.table(USER_TABLE)
        .select("id, login_id, password_hash")
        .eq("login_id", login_id)
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        return False, "아이디 또는 비밀번호가 올바르지 않습니다.", None
    row = rows[0]
    if not verify_password(password, row["password_hash"]):
        return False, "아이디 또는 비밀번호가 올바르지 않습니다.", None
    return True, "로그인되었습니다.", str(row["id"])


def _format_memory_block(messages: list[dict[str, str]], max_items: int = 50) -> str:
    tail = messages[-max_items:] if len(messages) > max_items else messages
    lines: list[str] = []
    for m in tail:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        prefix = "사용자" if role == "user" else "어시스턴트"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def _build_rag_messages(
    question: str,
    context: str,
    memory_text: str,
) -> list[SystemMessage | HumanMessage]:
    sys = f"""{ANSWER_STYLE_SYSTEM}

아래 [대화 맥락]과 [참고 문서]를 활용해 답하세요. 참고 문서에 없는 내용은 추측하지 말고 한계를 밝히세요.
[대화 맥락]
{memory_text or "(없음)"}

[참고 문서]
{context}
"""
    return [SystemMessage(content=sys), HumanMessage(content=question)]


def _generate_followup_section(llm: ChatOpenAI, user_q: str, answer: str) -> str:
    trimmed = answer[:8000]
    prompt = (
        "다음 사용자 질문과 답변을 바탕으로, 이어서 물어볼 만한 후속 질문을 한국어로 정확히 3개만 작성하세요.\n"
        "형식:\n1. ...\n2. ...\n3. ...\n"
        "설명 문장이나 다른 텍스트는 출력하지 마세요.\n\n"
        f"[사용자 질문]\n{user_q}\n\n[답변]\n{trimmed}"
    )
    try:
        out = llm.invoke([HumanMessage(content=prompt)])
        raw = remove_separators(str(getattr(out, "content", out) or ""))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Follow-up generation failed: %s", exc)
        return ""
    if not raw.strip():
        return ""
    return f"\n\n### 💡 다음에 물어볼 수 있는 질문들\n\n{raw.strip()}\n"


def _generate_session_title(llm: ChatOpenAI, first_q: str, first_a: str) -> str:
    prompt = (
        "다음 첫 질문과 답변을 한 줄로 요약한 대화 세션 제목을 한국어 30자 이내로만 출력하세요.\n"
        "따옴표, 설명, 번호 없이 제목만 출력하세요.\n\n"
        f"[질문]\n{first_q[:2000]}\n\n[답변]\n{first_a[:2000]}"
    )
    try:
        out = llm.invoke([HumanMessage(content=prompt)])
        title = str(getattr(out, "content", out) or "").strip()
        title = title.strip("\"'").replace("\n", " ")
        return title[:80] if title else "새 대화"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Session title generation failed: %s", exc)
        return (first_q[:40] + "…") if len(first_q) > 40 else (first_q or "새 대화")


# ---------------------------------------------------------------------------
# Supabase helpers (항상 user_id 필터)
# ---------------------------------------------------------------------------
def _assert_session_owner(sb: Client, session_id: str, user_id: str) -> None:
    resp = (
        sb.table("chat_sessions")
        .select("id")
        .eq("id", session_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    if not resp.data:
        raise PermissionError("이 세션에 접근할 권한이 없습니다.")


def fetch_all_sessions(sb: Client, user_id: str) -> list[dict[str, Any]]:
    resp = (
        sb.table("chat_sessions")
        .select("id, title, created_at, updated_at")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .execute()
    )
    return list(resp.data or [])


def fetch_session_messages(sb: Client, session_id: str, user_id: str) -> list[dict[str, Any]]:
    _assert_session_owner(sb, session_id, user_id)
    resp = (
        sb.table("chat_messages")
        .select("role, content, message_order")
        .eq("session_id", session_id)
        .eq("user_id", user_id)
        .order("message_order")
        .execute()
    )
    return list(resp.data or [])


def fetch_vector_file_names(sb: Client, session_id: str, user_id: str) -> list[str]:
    _assert_session_owner(sb, session_id, user_id)
    resp = (
        sb.table("vector_documents")
        .select("file_name")
        .eq("session_id", session_id)
        .execute()
    )
    names = sorted({row["file_name"] for row in (resp.data or []) if row.get("file_name")})
    return names


def delete_session_from_db(sb: Client, session_id: str, user_id: str) -> None:
    _assert_session_owner(sb, session_id, user_id)
    sb.table("chat_sessions").delete().eq("id", session_id).eq("user_id", user_id).execute()


def insert_session(sb: Client, title: str, user_id: str) -> str:
    resp = (
        sb.table("chat_sessions")
        .insert({"title": title, "user_id": user_id})
        .execute()
    )
    row = (resp.data or [{}])[0]
    return str(row["id"])


def replace_session_messages(
    sb: Client,
    session_id: str,
    user_id: str,
    messages: list[dict[str, str]],
) -> None:
    _assert_session_owner(sb, session_id, user_id)
    sb.table("chat_messages").delete().eq("session_id", session_id).eq("user_id", user_id).execute()
    rows = [
        {
            "session_id": session_id,
            "user_id": user_id,
            "role": m["role"],
            "content": m["content"],
            "message_order": idx,
        }
        for idx, m in enumerate(messages)
    ]
    if rows:
        sb.table("chat_messages").insert(rows).execute()


def update_session_title(sb: Client, session_id: str, user_id: str, title: str) -> None:
    sb.table("chat_sessions").update({"title": title}).eq("id", session_id).eq("user_id", user_id).execute()


def copy_vectors_to_session(
    sb: Client,
    source_session_id: str,
    target_session_id: str,
    user_id: str,
) -> None:
    _assert_session_owner(sb, source_session_id, user_id)
    _assert_session_owner(sb, target_session_id, user_id)
    resp = (
        sb.table("vector_documents")
        .select("file_name, content, metadata, embedding")
        .eq("session_id", source_session_id)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        return
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(
            {
                "session_id": target_session_id,
                "file_name": row["file_name"],
                "content": row["content"],
                "metadata": row.get("metadata") or {},
                "embedding": row["embedding"],
            }
        )
        if len(batch) >= VECTOR_BATCH_SIZE:
            sb.table("vector_documents").insert(batch).execute()
            batch = []
    if batch:
        sb.table("vector_documents").insert(batch).execute()


def store_vectors_for_session(
    sb: Client,
    embeddings: OpenAIEmbeddings,
    session_id: str,
    user_id: str,
    uploaded_files: list[Any],
) -> list[str]:
    _assert_session_owner(sb, session_id, user_id)
    if not uploaded_files:
        return []

    processed_names: list[str] = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    for uf in uploaded_files:
        file_name = Path(uf.name).name
        suffix = Path(uf.name).suffix.lower() or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.getvalue())
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            raw_docs = loader.load()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        if not raw_docs:
            continue

        for doc in raw_docs:
            doc.metadata["file_name"] = file_name

        splits = splitter.split_documents(raw_docs)
        if not splits:
            continue

        texts = [d.page_content for d in splits]
        metas = [dict(d.metadata) for d in splits]

        for i in range(0, len(texts), VECTOR_BATCH_SIZE):
            batch_texts = texts[i : i + VECTOR_BATCH_SIZE]
            batch_metas = metas[i : i + VECTOR_BATCH_SIZE]
            vectors = embeddings.embed_documents(batch_texts)
            rows = [
                {
                    "session_id": session_id,
                    "file_name": file_name,
                    "content": text,
                    "metadata": meta,
                    "embedding": vec,
                }
                for text, meta, vec in zip(batch_texts, batch_metas, vectors, strict=True)
            ]
            sb.table("vector_documents").insert(rows).execute()

        processed_names.append(file_name)

    return processed_names


def match_documents_rpc(
    sb: Client,
    embeddings: OpenAIEmbeddings,
    session_id: str,
    user_id: str,
    query: str,
    k: int = 10,
) -> list[Document]:
    _assert_session_owner(sb, session_id, user_id)
    query_vec = embeddings.embed_query(query)
    try:
        resp = sb.rpc(
            "match_vector_documents",
            {
                "query_embedding": query_vec,
                "match_count": k,
                "filter_session_id": session_id,
            },
        ).execute()
        docs: list[Document] = []
        for row in resp.data or []:
            docs.append(
                Document(
                    page_content=row.get("content") or "",
                    metadata={
                        "file_name": row.get("file_name"),
                        **(row.get("metadata") or {}),
                    },
                )
            )
        return docs
    except Exception as exc:  # noqa: BLE001
        logger.warning("RPC match_vector_documents failed: %s", exc)
        return _fallback_vector_search(sb, session_id, user_id, query, k)


def _fallback_vector_search(
    sb: Client,
    session_id: str,
    user_id: str,
    query: str,
    k: int,
) -> list[Document]:
    _assert_session_owner(sb, session_id, user_id)
    resp = (
        sb.table("vector_documents")
        .select("content, metadata, file_name")
        .eq("session_id", session_id)
        .limit(k * 3)
        .execute()
    )
    docs: list[Document] = []
    q_lower = query.lower()
    for row in resp.data or []:
        content = row.get("content") or ""
        if q_lower in content.lower() or not q_lower:
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "file_name": row.get("file_name"),
                        **(row.get("metadata") or {}),
                    },
                )
            )
        if len(docs) >= k:
            break
    if not docs and resp.data:
        for row in resp.data[:k]:
            docs.append(
                Document(
                    page_content=row.get("content") or "",
                    metadata={"file_name": row.get("file_name")},
                )
            )
    return docs


def auto_save_session(
    sb: Client,
    llm: ChatOpenAI | None,
    session_id: str,
    user_id: str,
    chat_history: list[dict[str, str]],
) -> None:
    replace_session_messages(sb, session_id, user_id, chat_history)
    user_msgs = [m for m in chat_history if m["role"] == "user"]
    asst_msgs = [m for m in chat_history if m["role"] == "assistant"]
    if llm and user_msgs and asst_msgs:
        title = _generate_session_title(llm, user_msgs[0]["content"], asst_msgs[0]["content"])
        update_session_title(sb, session_id, user_id, title)


def load_session_into_state(sb: Client, session_id: str, user_id: str) -> None:
    messages = fetch_session_messages(sb, session_id, user_id)
    st.session_state.current_session_id = session_id
    st.session_state.chat_history = [
        {"role": m["role"], "content": m["content"]} for m in messages
    ]
    st.session_state.conversation_memory = list(st.session_state.chat_history)
    st.session_state.processed_names = fetch_vector_file_names(sb, session_id, user_id)
    sessions = fetch_all_sessions(sb, user_id)
    st.session_state.session_options = {
        f"{s['title']} ({str(s['id'])[:8]}…)": str(s["id"]) for s in sessions
    }


def _init_session_state() -> None:
    defaults = {
        "chat_history": [],
        "conversation_memory": [],
        "processed_names": [],
        "current_session_id": None,
        "session_options": {},
        "selected_session_label": None,
        "keys_ok": True,
        "sb": None,
        "openai_key": "",
        "user_id": None,
        "login_id": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _clear_ui_state(keep_sb: bool = True) -> None:
    st.session_state.chat_history = []
    st.session_state.conversation_memory = []
    st.session_state.processed_names = []
    st.session_state.current_session_id = None
    if not keep_sb:
        st.session_state.sb = None


def _logout() -> None:
    st.session_state.user_id = None
    st.session_state.login_id = None
    _clear_ui_state(keep_sb=True)
    st.session_state.session_options = {}
    st.session_state.selected_session_label = None


def _ensure_working_session(sb: Client, user_id: str) -> str:
    if st.session_state.current_session_id:
        sid = str(st.session_state.current_session_id)
        try:
            _assert_session_owner(sb, sid, user_id)
            return sid
        except PermissionError:
            st.session_state.current_session_id = None
    new_id = insert_session(sb, "새 대화", user_id)
    st.session_state.current_session_id = new_id
    return new_id


def _render_header() -> None:
    st.markdown(
        """
<style>
h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
div.stButton > button:first-child {
  background-color: #ff69b4;
  color: #ffffff;
}
div.stButton > button:hover {
  background-color: #ff1493;
  color: #ffffff;
}
</style>
""",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([1, 4, 1])
    with c1:
        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=180)
        else:
            st.markdown("### 📚")
    with c2:
        st.markdown(
            """
<h1 style="text-align:center; margin:0;">
  <span style="color:#1f77b4;">재정경제부</span>
  <span style="color:#ff8c00;">RAG 챗봇</span>
</h1>
""",
            unsafe_allow_html=True,
        )
    with c3:
        st.empty()


def _refresh_session_dropdown(sb: Client, user_id: str) -> None:
    sessions = fetch_all_sessions(sb, user_id)
    st.session_state.session_options = {
        f"{s['title']} ({str(s['id'])[:8]}…)": str(s["id"]) for s in sessions
    }


def _render_auth_screen(sb: Client) -> None:
    st.markdown("### 로그인 / 회원가입")
    tab_login, tab_signup = st.tabs(["로그인", "회원가입"])

    with tab_login:
        login_id = st.text_input("아이디", key="auth_login_id")
        password = st.text_input("비밀번호", type="password", key="auth_login_pw")
        if st.button("로그인", use_container_width=True, key="btn_login"):
            ok, msg, uid = login_user(sb, login_id, password)
            if ok and uid:
                st.session_state.user_id = uid
                st.session_state.login_id = login_id.strip()
                _clear_ui_state()
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    with tab_signup:
        new_id = st.text_input("아이디", key="auth_signup_id")
        new_pw = st.text_input("비밀번호", type="password", key="auth_signup_pw")
        new_pw2 = st.text_input("비밀번호 확인", type="password", key="auth_signup_pw2")
        if st.button("회원가입", use_container_width=True, key="btn_signup"):
            if new_pw != new_pw2:
                st.error("비밀번호가 일치하지 않습니다.")
            else:
                ok, msg = register_user(sb, new_id, new_pw)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)


def _render_chat_app(sb: Client, user_id: str) -> None:
    llm = get_llm(st.session_state.openai_key)
    embeddings = get_embeddings(st.session_state.openai_key)
    _refresh_session_dropdown(sb, user_id)

    with st.sidebar:
        st.markdown(f"**로그인:** `{st.session_state.login_id}`")
        if st.button("로그아웃", use_container_width=True):
            _logout()
            st.rerun()

        st.markdown("---")
        st.markdown("**LLM 모델**")
        st.info(f"고정 모델: `{MODEL_NAME}`")

        rag_choice = st.radio(
            "RAG (PDF 검색) 선택",
            ("사용 안 함", "RAG 사용"),
            index=0,
        )

        uploads = st.file_uploader(
            "PDF 파일 업로드",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if st.button("파일 처리하기", use_container_width=True):
            if not uploads:
                st.warning("업로드된 PDF가 없습니다.")
            else:
                try:
                    sid = _ensure_working_session(sb, user_id)
                    sb.table("vector_documents").delete().eq("session_id", sid).execute()
                    names = store_vectors_for_session(
                        sb, embeddings, sid, user_id, list(uploads)
                    )
                    st.session_state.processed_names = names
                    auto_save_session(sb, llm, sid, user_id, st.session_state.chat_history)
                    _refresh_session_dropdown(sb, user_id)
                    st.success("PDF 처리 및 세션 자동 저장이 완료되었습니다.")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("PDF 처리 실패: %s", exc)
                    st.error(f"PDF 처리 중 오류: {exc}")

        if st.session_state.processed_names:
            st.markdown("**처리된 파일**")
            for name in st.session_state.processed_names:
                st.text(f"- {name}")

        st.markdown("---")
        st.markdown("**세션 관리**")

        labels = list(st.session_state.session_options.keys())
        if labels:

            def _on_session_select() -> None:
                label = st.session_state.session_selectbox
                if label and label in st.session_state.session_options:
                    load_session_into_state(
                        st.session_state.sb,
                        st.session_state.session_options[label],
                        st.session_state.user_id,
                    )

            default_idx = 0
            if st.session_state.selected_session_label in labels:
                default_idx = labels.index(st.session_state.selected_session_label)
            chosen_label = st.selectbox(
                "저장된 세션",
                labels,
                index=default_idx,
                key="session_selectbox",
                on_change=_on_session_select,
            )
            st.session_state.selected_session_label = chosen_label
        else:
            st.caption("저장된 세션이 없습니다.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("세션저장", use_container_width=True):
                if not st.session_state.chat_history:
                    st.warning("저장할 대화가 없습니다.")
                else:
                    try:
                        user_msgs = [
                            m for m in st.session_state.chat_history if m["role"] == "user"
                        ]
                        asst_msgs = [
                            m
                            for m in st.session_state.chat_history
                            if m["role"] == "assistant"
                        ]
                        if user_msgs and asst_msgs:
                            title = _generate_session_title(
                                llm, user_msgs[0]["content"], asst_msgs[0]["content"]
                            )
                        else:
                            title = "새 대화"
                        new_id = insert_session(sb, title, user_id)
                        replace_session_messages(
                            sb, new_id, user_id, st.session_state.chat_history
                        )
                        src = st.session_state.current_session_id
                        if src:
                            copy_vectors_to_session(sb, str(src), new_id, user_id)
                        st.session_state.current_session_id = new_id
                        _refresh_session_dropdown(sb, user_id)
                        st.success(f"세션이 저장되었습니다: {title}")
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"세션 저장 실패: {exc}")

            if st.button("세션로드", use_container_width=True):
                label = st.session_state.get("selected_session_label")
                if not label or label not in st.session_state.session_options:
                    st.warning("로드할 세션을 선택하세요.")
                else:
                    sid = st.session_state.session_options[label]
                    load_session_into_state(sb, sid, user_id)
                    st.success("세션을 불러왔습니다.")

        with col2:
            if st.button("세션삭제", use_container_width=True):
                sid = st.session_state.current_session_id
                if not sid:
                    st.warning("삭제할 세션이 없습니다.")
                else:
                    try:
                        delete_session_from_db(sb, str(sid), user_id)
                        _clear_ui_state()
                        _refresh_session_dropdown(sb, user_id)
                        st.success("세션이 삭제되었습니다.")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"세션 삭제 실패: {exc}")

            if st.button("화면초기화", use_container_width=True):
                _clear_ui_state()
                st.rerun()

        if st.button("vectordb", use_container_width=True):
            sid = st.session_state.current_session_id
            if not sid:
                st.warning("현재 활성 세션이 없습니다.")
            else:
                try:
                    names = fetch_vector_file_names(sb, str(sid), user_id)
                    if names:
                        st.markdown("**Vector DB 파일 목록**")
                        for n in names:
                            st.text(f"- {n}")
                    else:
                        st.info("이 세션에 저장된 벡터 문서가 없습니다.")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

        mem_count = len(st.session_state.conversation_memory)
        file_count = len(st.session_state.processed_names)
        sid_short = (
            str(st.session_state.current_session_id)[:8] + "…"
            if st.session_state.current_session_id
            else "없음"
        )
        st.text(
            f"모델: {MODEL_NAME}\n"
            f"RAG: {rag_choice}\n"
            f"현재 세션: {sid_short}\n"
            f"처리된 PDF: {file_count}\n"
            f"대화 메시지 수: {mem_count}"
        )

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(remove_separators(msg["content"]))

    user_input = st.chat_input("질문을 입력하세요")
    if not user_input:
        return

    sid = _ensure_working_session(sb, user_id)

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.conversation_memory.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(remove_separators(user_input))

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""

        try:
            if rag_choice == "RAG 사용":
                has_vectors = bool(fetch_vector_file_names(sb, sid, user_id))
                if not has_vectors:
                    full_answer = (
                        "# 안내\n\n"
                        "RAG를 사용하려면 PDF를 업로드한 뒤 **파일 처리하기**를 눌러 주세요."
                    )
                    placeholder.markdown(remove_separators(full_answer))
                else:
                    mem_txt = _format_memory_block(
                        st.session_state.conversation_memory[:-1]
                    )
                    docs = match_documents_rpc(sb, embeddings, sid, user_id, user_input, k=10)
                    context = "\n\n".join(d.page_content for d in docs)
                    messages = _build_rag_messages(user_input, context, mem_txt)
                    acc = ""
                    for chunk in llm.stream(messages):
                        piece = getattr(chunk, "content", "") or ""
                        if piece:
                            acc += piece
                            placeholder.markdown(remove_separators(acc) + "▌")
                    full_answer = remove_separators(acc)
                    placeholder.markdown(full_answer)
            else:
                mem_txt = _format_memory_block(
                    st.session_state.conversation_memory[:-1]
                )
                sys = f"{ANSWER_STYLE_SYSTEM}\n\n[대화 맥락]\n{mem_txt or '(없음)'}"
                msgs = [
                    SystemMessage(content=sys),
                    HumanMessage(content=user_input),
                ]
                acc = ""
                for chunk in llm.stream(msgs):
                    piece = getattr(chunk, "content", "") or ""
                    if piece:
                        acc += piece
                        placeholder.markdown(remove_separators(acc) + "▌")
                full_answer = remove_separators(acc)
                placeholder.markdown(full_answer)

            if full_answer and not full_answer.lstrip().startswith("# 오류"):
                follow = _generate_followup_section(llm, user_input, full_answer)
                if follow:
                    full_answer += follow
                    placeholder.markdown(remove_separators(full_answer))

        except Exception as exc:  # noqa: BLE001
            logger.warning("답변 생성 실패: %s", exc)
            full_answer = f"# 오류\n\n요청 처리 중 문제가 발생했습니다.\n\n`{exc}`"
            placeholder.markdown(remove_separators(full_answer))

    st.session_state.chat_history.append({"role": "assistant", "content": full_answer})
    st.session_state.conversation_memory.append(
        {"role": "assistant", "content": full_answer}
    )

    try:
        auto_save_session(sb, llm, sid, user_id, st.session_state.chat_history)
        _refresh_session_dropdown(sb, user_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("자동 저장 실패: %s", exc)


def main() -> None:
    st.set_page_config(
        page_title="재정경제부 RAG 챗봇",
        page_icon="📚",
        layout="wide",
    )
    _init_session_state()
    _render_header()

    keys = _env_keys()
    missing = _missing_keys(keys)
    if missing:
        st.error(
            "다음 키가 설정되지 않았습니다: **"
            + ", ".join(missing)
            + "**\n\n"
            "Streamlit Cloud에서는 **Secrets**에, 로컬에서는 **`.env`**에 설정하세요.\n\n"
            f"로컬 `.env` 경로: `{ENV_PATH}`"
        )
        st.stop()

    st.session_state.openai_key = keys["openai"]
    if st.session_state.sb is None:
        st.session_state.sb = get_supabase_client(keys)
    sb: Client = st.session_state.sb

    if not st.session_state.user_id:
        _render_auth_screen(sb)
        return

    _render_chat_app(sb, str(st.session_state.user_id))


if __name__ == "__main__":
    main()
