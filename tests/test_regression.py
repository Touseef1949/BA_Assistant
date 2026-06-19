import app


def test_simplified_mode_contract_remains_locked():
    assert app.ANALYSIS_TYPES == ["Standard", "Interactive (Q&A)", "Deep Team"]
    assert "Quick Feature Extraction" not in app.ANALYSIS_TYPES
    assert "User Stories Generation" not in app.ANALYSIS_TYPES


def test_deepseek_flash_is_enforced_for_text_model_paths():
    assert app.TEXT_ANALYSIS_MODEL_ID == "deepseek-v4-flash"
    assert app.DEEPSEEK_BASE_URL == "https://api.deepseek.com/v1"
    assert app.AppConfig(
        project_name="P",
        analysis_type="Standard",
        model_id=app.TEXT_ANALYSIS_MODEL_ID,
        render_mermaid=True,
        mermaid_theme="neutral",
        add_confetti=False,
        show_prompt_preview=False,
        show_member_responses=False,
    ).model_id == "deepseek-v4-flash"


def test_markdown_export_filename_slug_invariant():
    config = app.AppConfig(
        project_name="Payments & KYC MVP!",
        analysis_type="Standard",
        model_id=app.TEXT_ANALYSIS_MODEL_ID,
        render_mermaid=True,
        mermaid_theme="neutral",
        add_confetti=False,
        show_prompt_preview=False,
        show_member_responses=False,
    )
    result = "# Report\n\n```mermaid\nflowchart TD\nA-->B\n```"

    pdf = app.generate_pdf(config.project_name, result)
    mermaid = app.extract_mermaid_code(result)

    assert pdf.startswith(b"%PDF")
    assert mermaid.startswith("flowchart TD")
