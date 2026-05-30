"""
Thin CLI wrapper to run the BA Assistant requirement analysis engine directly,
bypassing Streamlit. Uses the same RequirementAnalyzer class + Agno multi-agent team.

Usage: python run_analysis_cli.py "Your raw requirements here"
"""
import sys
import os

# Point to the BA_Assistant directory so imports work
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import dotenv
dotenv.load_dotenv(".env")

from deepseek_requirement_analysis_app import RequirementAnalyzer, build_analysis_prompt, AppConfig, event_to_text

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_analysis_cli.py \"<requirements text>\"")
        sys.exit(1)

    requirements = sys.argv[1]

    config = AppConfig(
        project_name="food-delivery-app",
        analysis_type="Comprehensive",
        model_id="deepseek-v4-flash",
        render_mermaid=True,
        mermaid_theme="neutral",
        add_confetti=False,
        show_prompt_preview=False,
        show_member_responses=False,
    )

    print("=" * 70)
    print("  BA ASSISTANT — Multi-Agent Requirement Analysis")
    print("=" * 70)
    print(f"\nInput requirements:\n  {requirements[:200]}{'...' if len(requirements) > 200 else ''}\n")
    print("Running analysis with 5-agent team (BA, Product Owner, Architect, Risk/QA, Diagram)...\n")
    print("-" * 70)

    analyzer = RequirementAnalyzer(
        model_id="deepseek-v4-flash",
        show_member_responses=False,
    )

    stream = analyzer.run_analysis(requirements, config)

    full_output = []
    for event in stream:
        text = event_to_text(event)
        if text:
            print(text, end="", flush=True)
            full_output.append(text)

    print("\n" + "=" * 70)
    print("  Analysis complete.")
    print("=" * 70)

    # Also save to a markdown file
    output_path = "analysis_output.md"
    with open(output_path, "w") as f:
        f.write("".join(full_output))
    print(f"\nFull report saved to: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
