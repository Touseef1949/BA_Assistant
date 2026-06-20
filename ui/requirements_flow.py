"""Streamlit requirements input and interactive Q&A flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import streamlit as st
from PIL import Image as PILImage


@dataclass(frozen=True)
class RequirementsFlowDependencies:
    financial_templates: Dict[str, Tuple[str, str]]
    report_structure: str
    extract_pdf_text_fn: Callable[[Any], str]
    require_runtime_dependencies_fn: Callable[[bool], bool]
    require_api_keys_fn: Callable[[bool], bool]
    analyzer_factory: Callable[..., Any]
    reset_interactive_fn: Callable[[], None]
    run_paid_gate_fn: Callable[[str, bool], bool]
    parse_questions_fn: Callable[[str], List[str]]
    stream_to_markdown_fn: Callable[[Callable[[bool], Any], Any], str]
    extract_mermaid_code_fn: Callable[[str], str]
    save_history_fn: Callable[..., List[Dict[str, Any]]]
    safe_secret_fn: Callable[[str, str], str]


def render_template_selector(deps: RequirementsFlowDependencies) -> None:
    template_keys = list(deps.financial_templates.keys())
    labels = [deps.financial_templates[key][0] for key in template_keys]
    current = st.session_state.get("selected_template", "loan_origination")
    index = template_keys.index(current) if current in template_keys else 1
    selected_label = st.selectbox("📋 Choose Template", labels, index=index)
    selected_key = template_keys[labels.index(selected_label)]
    st.session_state["selected_template"] = selected_key

    if st.session_state.get("_last_template") != selected_key:
        st.session_state["_last_template"] = selected_key
        if selected_key != "none":
            st.session_state["requirements_area"] = deps.financial_templates[selected_key][1]
        deps.reset_interactive_fn()
        st.rerun()


def render_upload_area(config: Any, deps: RequirementsFlowDependencies) -> None:
    uploaded_file = st.file_uploader(
        "📎 Upload a document (optional) — PRD, email, whiteboard photo, or PDF",
        type=["png", "jpg", "jpeg", "pdf"],
        help="Upload a document to extract requirements from. Works best with clear text/screenshots.",
    )

    if not uploaded_file:
        return

    signature = f"{uploaded_file.name}:{uploaded_file.size}:{uploaded_file.type}"
    if uploaded_file.type == "application/pdf":
        if st.session_state.get("last_uploaded_signature") != signature:
            with st.spinner("Extracting PDF text with pdfplumber..."):
                extracted_text = deps.extract_pdf_text_fn(uploaded_file)
            st.session_state["requirements_area"] = extracted_text
            st.session_state["last_uploaded_signature"] = signature
            deps.reset_interactive_fn()
            st.success("PDF text extracted into the requirements area.")
            st.rerun()
        else:
            st.info("PDF already extracted into the requirements area.")
        return

    try:
        uploaded_file.seek(0)
        st.image(PILImage.open(uploaded_file), caption="Uploaded document", use_container_width=True)
        uploaded_file.seek(0)
    except Exception:
        st.image(uploaded_file, caption="Uploaded document", use_container_width=True)

    if st.button("🔍 Extract Requirements from Image", type="secondary"):
        if not deps.require_runtime_dependencies_fn(vision=True) or not deps.require_api_keys_fn(vision=True):
            return
        image_bytes = uploaded_file.getvalue()
        with st.spinner("Analyzing document with Gemini Vision..."):
            analyzer = deps.analyzer_factory(config.model_id, config.show_member_responses, enable_vision=True)
            extracted = analyzer.extract_requirements_from_image(image_bytes, uploaded_file.type)
        st.session_state["requirements_area"] = extracted
        st.session_state["last_uploaded_signature"] = signature
        deps.reset_interactive_fn()
        st.success("Image requirements extracted into the requirements area.")
        st.rerun()


def render_prompt_preview(config: Any, requirements_text: str, deps: RequirementsFlowDependencies) -> None:
    if not config.show_prompt_preview:
        return
    with st.expander("Prompt preview", expanded=False):
        preview_analyzer = None
        try:
            if deps.require_runtime_dependencies_fn(False):
                preview_analyzer = deps.analyzer_factory(config.model_id, config.show_member_responses, enable_vision=False)
        except Exception:
            preview_analyzer = None
        if preview_analyzer:
            st.code(preview_analyzer.compose_prompt(requirements_text, config.project_name, config.analysis_type), language="markdown")
        else:
            st.code(f"{deps.report_structure}\n\nRequirements:\n{requirements_text}", language="markdown")


def render_interactive_flow(config: Any, email: str, requirements_text: str, deps: RequirementsFlowDependencies) -> None:
    st.markdown("#### Interactive Q&A mode")
    st.caption("Step 1: generate clarifying questions (single agent) · Step 2: answer them · Step 3: run the multi-agent Team with your answers as extra context.")

    stage = st.session_state.get("interactive_stage", "input")
    if stage == "input":
        if st.button("🔍 Analyze & Generate Questions", type="primary", use_container_width=True):
            if not requirements_text.strip():
                st.warning("Add requirements or upload a document first.")
                return
            if not deps.run_paid_gate_fn(email, consume_usage=False):
                return
            if not deps.require_runtime_dependencies_fn(False) or not deps.require_api_keys_fn(False):
                return
            with st.spinner("Analyzing requirements and generating clarifying questions..."):
                analyzer = deps.analyzer_factory(config.model_id, config.show_member_responses, enable_vision=False)
                raw_questions = analyzer.generate_questions(requirements_text)
            st.session_state["interactive_questions"] = deps.parse_questions_fn(raw_questions)
            st.session_state["interactive_answers"] = {}
            st.session_state["interactive_stage"] = "questions"
            st.rerun()
        return

    if stage == "questions":
        questions = st.session_state.get("interactive_questions", [])
        if not questions:
            st.session_state["interactive_stage"] = "input"
            st.rerun()

        st.info("Answer the clarifying questions below. Empty answers are allowed but will be marked as unknown.")
        for i, question in enumerate(questions):
            key = f"interactive_answer_{i}"
            current = st.session_state.get("interactive_answers", {}).get(question, "")
            answer = st.text_input(f"Q{i + 1}: {question}", value=current, key=key)
            st.session_state["interactive_answers"][question] = answer

        col_a, col_b = st.columns([2, 1])
        with col_a:
            generate_clicked = st.button("✅ Generate Full Report", type="primary", use_container_width=True)
        with col_b:
            if st.button("↩️ Restart Q&A", use_container_width=True):
                deps.reset_interactive_fn()
                st.rerun()

        if generate_clicked:
            if not deps.run_paid_gate_fn(email, consume_usage=True):
                return
            if not deps.require_runtime_dependencies_fn(False) or not deps.require_api_keys_fn(False):
                return
            qa_transcript = "\n".join(
                f"Q: {q}\nA: {a.strip() or '[Unknown / not answered]'}"
                for q, a in st.session_state["interactive_answers"].items()
            )
            st.session_state["interactive_stage"] = "generate"
            placeholder = st.empty()
            with st.spinner("Running multi-agent Team with enriched Q&A context..."):
                analyzer = deps.analyzer_factory(config.model_id, config.show_member_responses, enable_vision=False)
                result = deps.stream_to_markdown_fn(
                    lambda stream: analyzer.run_interactive(requirements_text, config.project_name, qa_transcript, stream=stream),
                    placeholder,
                )
            st.session_state["last_result"] = result
            st.session_state["last_mermaid"] = deps.extract_mermaid_code_fn(result)
            st.session_state["history"] = deps.save_history_fn(
                config.project_name,
                config.analysis_type,
                result,
                st.session_state.get("history", []),
                deps.safe_secret_fn,
                email=email,
            )
            if config.add_confetti:
                st.balloons()
            st.success("Interactive report generated.")
            st.session_state["interactive_stage"] = "questions"
