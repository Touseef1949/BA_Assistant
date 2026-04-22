import streamlit as st
from agno.agent import Agent, RunContentEvent
from agno.models.groq import Groq
from agno.team.team import Team
from agno.tools.reasoning import ReasoningTools
import os
import dotenv
from typing import Generator, List, Dict
import time
from datetime import datetime

# =============================
# Environment bootstrap
# =============================
try:
    dotenv.load_dotenv()
except Exception as e:
    print(f"Error loading .env file (this is expected if running in a deployed environment): {e}")

# =============================
# Custom visual helpers (no external deps)
# =============================
CARD_CSS = """
<style>
/****** App polish ******/
:root { --brand-gradient: linear-gradient(90deg, #7c3aed, #06b6d4, #22c55e); }
.stApp header { backdrop-filter: blur(6px); }

h1 span.gradient, .gradient-text { 
  background: var(--brand-gradient);
  -webkit-background-clip: text; background-clip: text; color: transparent; 
}

/* Glass cards */
.block-container { padding-top: 2rem !important; }
.card {
  border-radius: 18px; padding: 18px 18px; 
  border: 1px solid rgba(120, 120, 120, .15);
  background: rgba(255,255,255,.6);
  box-shadow: 0 10px 30px rgba(0,0,0,.06);
}
[data-base-theme="dark"] .card {
  background: rgba(13, 17, 23, .55);
  border-color: rgba(255,255,255,.08);
  box-shadow: 0 10px 30px rgba(0,0,0,.35);
}

/* Fancy primary button */
button[kind="primary"] { position: relative; overflow: hidden; }
button[kind="primary"]:before { 
  content:""; position:absolute; inset: -2px; z-index: -1; border-radius: 12px; 
  background: var(--brand-gradient); filter: blur(6px); opacity:.7;
}

/* Code blocks */
.code-block { 
  border-radius: 12px; border: 1px solid rgba(120,120,120,.2); padding: 12px; 
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
}

.badge { 
  display:inline-flex; align-items:center; gap:.45rem; padding:.25rem .6rem; 
  border-radius: 999px; border:1px solid rgba(120,120,120,.2);
}
.badge .dot { width:.5rem; height:.5rem; border-radius:999px; background:#22c55e; box-shadow:0 0 6px #22c55eAA }

.step { display:flex; align-items:flex-start; gap:.75rem; margin:.25rem 0; }
.step .num { 
  width:1.6rem; height:1.6rem; border-radius:999px; 
  background: #7c3aed22; color:#7c3aed; font-weight:700; display:flex; align-items:center; justify-content:center;
  border:1px solid #7c3aed44;
}
</style>
"""

# =============================
# Diagram generation tool (given)
# =============================
class DiagramGenerationTools:
    def generate_mermaid_diagram(self, requirements: str, flow_type: str = "flowchart") -> str:
        if flow_type.lower() == "flowchart":
            mermaid_code = f"""
graph TD
    A[Start] --> B{{User visits website}};
    B --> C{{Browse products}};
    C --> D{{Select a product}};
    D --> E[Add to cart];
    E --> F{{View cart}};
    F --> G{{Proceed to checkout}};
    G --> H[Enter shipping details];
    H --> I[Select payment method];
    I --> J{{Confirm and pay}};
    J --> K[Process payment];
    K -- Success --> L[Send order confirmation email];
    K -- Failure --> M[Show payment failed message];
    L --> N[Admin manages inventory];
    L --> O[User tracks order];
    O --> P[End];
    M --> G;
"""
            return mermaid_code
        elif flow_type.lower() == "sequence":
            return "sequenceDiagram\n    participant User\n    participant System\n    User->>System: Browse products\n    System-->>User: Display products"
        else:
            return f"graph TD\n    A[Diagram type '{flow_type}' not supported in this example.]"

# =============================
# Agents (kept as provided)
# =============================
# Agent 1: Requirements Parser Agent
requirements_parser_agent = Agent(
    name="Requirements Parser Agent",
    role="Natural Language Processing specialist for parsing and structuring user requirements",
    model=Groq(id="openai/gpt-oss-120b"),
    instructions=[
        "Parse natural language requirements into structured components",
        "Identify key entities: actors, actions, objects, constraints",
        "Extract business rules and functional requirements",
        "Categorize requirements by type (functional, non-functional, business)",
        "Output structured JSON format with parsed components",
        "Handle ambiguous requirements by asking clarifying questions",
    ],
)

# Agent 2: Feature Extraction Agent
feature_extraction_agent = Agent(
    name="Feature Extraction Agent",
    role="Software architecture specialist focused on converting requirements into technical features",
    model=Groq(id="openai/gpt-oss-120b"),
    instructions=[
        "Convert parsed requirements into concrete software features",
        "Prioritize features using MoSCoW method (Must, Should, Could, Won't)",
        "Group related functionalities into feature sets",
        "Identify feature dependencies and relationships",
        "Estimate feature complexity (low, medium, high)",
        "Provide detailed feature descriptions with technical considerations",
        "Output features in JSON format with priority, description, and dependencies",
    ],
)

# Agent 3: User Story Generation Agent
user_story_agent = Agent(
    name="User Story Generation Agent",
    role="Agile methodology expert specializing in user story creation",
    model=Groq(id="openai/gpt-oss-120b"),
    instructions=[
        "Generate user stories following 'As a [user], I want [goal], so that [benefit]' format",
        "Create comprehensive acceptance criteria for each story using Given-When-Then format",
        "Ensure stories follow INVEST principles (Independent, Negotiable, Valuable, Estimable, Small, Testable)",
        "Map user stories to corresponding features",
        "Include edge cases and error scenarios in acceptance criteria",
        "Estimate story points using Fibonacci sequence",
        "Output in JSON format with stories, acceptance criteria, and estimates",
    ],
)

# Agent 4: Technical Architecture Agent
technical_architecture_agent = Agent(
    name="Technical Architecture Agent",
    role="Senior software architect specializing in technology recommendations and system design",
    model=Groq(id="openai/gpt-oss-120b"),
    instructions=[
        "Analyze technical requirements and recommend appropriate technology stack",
        "Consider scalability, maintainability, and performance requirements",
        "Suggest architectural patterns (MVC, microservices, etc.)",
        "Estimate development effort in hours/days",
        "Identify technical constraints and limitations",
        "Research current best practices and trending technologies",
        "Provide justification for technology choices",
        "Output comprehensive technical assessment in JSON format",
        "Include database design recommendations",
        "Consider security implications and requirements",
    ],
)

# Agent 5: Process Flow Designer Agent
process_flow_agent = Agent(
    name="Process Flow Designer Agent",
    role="Systems analyst and process modeling expert",
    model=Groq(id="openai/gpt-oss-120b"),
    instructions=[
        "Create detailed process flow diagrams using Mermaid syntax",
        "Design user journey workflows and system interactions",
        "Model both happy path and error handling flows",
        "Create different diagram types: flowcharts, sequence diagrams, state diagrams",
        "Ensure flows align with user stories and features",
        "Include decision points, alternative paths, and edge cases",
        "Generate clean, readable Mermaid code",
        "Provide flow descriptions and explanations",
        "Consider UI/UX flow implications",
    ],
)

# Agent 6: Risk Assessment Agent
risk_assessment_agent = Agent(
    name="Risk Assessment Agent",
    role="Project risk management and quality assurance specialist",
    model=Groq(id="openai/gpt-oss-120b"),
    tools=[ReasoningTools()],
    instructions=[
        "Identify technical, business, and project risks",
        "Assess risk probability and impact using standard risk matrices",
        "Propose concrete mitigation strategies for each identified risk",
        "Consider integration risks, scalability issues, and security vulnerabilities",
        "Research common pitfalls for similar projects",
        "Evaluate team skill requirements and knowledge gaps",
        "Assess timeline and budget risks",
        "Provide risk prioritization and mitigation roadmap",
        "Output structured risk assessment in JSON format",
        "Include contingency planning recommendations",
    ],
)

# Agent 7: Quality Assurance Agent
quality_assurance_agent = Agent(
    name="Quality Assurance Agent",
    role="Quality control specialist ensuring output consistency and completeness",
    model=Groq(id="llama-3.1-8b-versatile"),
    tools=[ReasoningTools()],
    instructions=[
        "Validate completeness and consistency across all agent outputs",
        "Check alignment between features, user stories, and technical architecture",
        "Ensure all requirements are addressed in the analysis",
        "Identify gaps, inconsistencies, or contradictions",
        "Verify that user stories match features and acceptance criteria are complete",
        "Validate technical recommendations against requirements",
        "Check process flows for logical consistency",
        "Provide quality score and improvement recommendations",
        "Ensure outputs follow specified JSON formats",
        "Consolidate and organize final deliverables",
    ],
)

# Agent 8: Project Coordinator Agent
project_coordinator_agent = Agent(
    name="Project Coordinator Agent",
    role="Project management and coordination specialist",
    model=Groq(id="openai/gpt-oss-120b"),
    instructions=[
        "Coordinate the analysis workflow and agent interactions",
        "Manage project metadata and organization",
        "Create executive summaries and project overviews",
        "Generate project timelines and milestones",
        "Coordinate between different analysis phases",
        "Handle project versioning and change management",
        "Create final consolidated reports",
        "Manage project documentation and deliverables",
        "Provide project status updates and completion tracking",
    ],
)

# --- CORRECTED Main Requirement Analysis Team ---
requirement_analysis_team = Team(
    name="Advanced Requirement Analysis Team",
    model=Groq(id="openai/gpt-oss-120b"),
    members=[
        requirements_parser_agent,
        feature_extraction_agent,
        user_story_agent,
        technical_architecture_agent,
        process_flow_agent,
        risk_assessment_agent,
        quality_assurance_agent,
        project_coordinator_agent,
    ],
    tools=[ReasoningTools(add_instructions=True)],
    instructions=[
        "Collaborate to provide comprehensive software requirement analysis",
        # --- THIS IS THE FIX ---
        "IMPORTANT: When delegating a task to a member, you MUST provide them with all the necessary context and data from the original user prompt. For the first step, this means passing the user's full requirements text to the Requirements Parser Agent.",
        # --- END OF FIX ---
        "Follow a structured workflow: Parse → Extract Features → Generate Stories → Assess Architecture → Design Flows → Evaluate Risks → Quality Check → Coordinate",
        "Ensure all outputs are consistent and aligned across agents",
        "Use JSON format for structured data exchange between agents",
        "Validate that all original requirements are addressed",
        "Present findings in a professional, structured format",
        "Include executive summary, detailed analysis, and actionable recommendations",
        "Only output the final consolidated analysis, not individual agent responses unless specifically requested",
        "Ensure traceability from requirements to features to user stories",
        "Provide clear next steps and implementation roadmap",
    ],
    markdown=True,
    show_members_responses=False,
)

# =============================
# Analyzer wrapper
# =============================
class RequirementAnalyzer:
    """A class to analyze software requirements using a team of AI agents."""
    def __init__(self):
        self.team = requirement_analysis_team

    def analyze_requirements(self, requirements_text: str, project_name: str, analysis_level: str) -> Generator[RunContentEvent, None, None]:
        analysis_instructions = {
            "Comprehensive": "Provide detailed analysis with full feature breakdown, user stories, and technical assessment",
            "Enterprise": "Provide enterprise-level analysis including detailed risk assessment, security considerations, and scalability planning",
        }

        prompt = f"""
        Analyze the following software requirements for the project: "{project_name or 'this project'}"

        REQUIREMENTS:
        {requirements_text}

        ANALYSIS LEVEL: {analysis_level}
        SPECIAL INSTRUCTIONS: {analysis_instructions.get(analysis_level, '')}

        Please provide a complete requirement analysis including:
        1. Parsed and structured requirements
        2. Prioritized feature list with complexity estimates
        3. User stories with acceptance criteria
        4. Technical architecture recommendations
        5. Process flow diagrams (Mermaid format)
        6. Risk assessment with mitigation strategies
        7. Implementation timeline and effort estimates
        8. Executive summary and next steps

        Format the output professionally with clear sections and actionable insights.
        """
        return self.team.run(prompt, stream=True)

    def quick_feature_extraction(self, requirements_text: str) -> Generator[RunContentEvent, None, None]:
        return feature_extraction_agent.run(
            f"Extract and prioritize features from these requirements: {requirements_text}",
            stream=True,
        )

    def generate_user_stories_only(self, requirements_text: str) -> Generator[RunContentEvent, None, None]:
        return user_story_agent.run(
            f"Create user stories with acceptance criteria for: {requirements_text}",
            stream=True,
        )

    def assess_technical_architecture(self, requirements_text: str) -> Generator[RunContentEvent, None, None]:
        return technical_architecture_agent.run(
            f"Provide technical architecture recommendations for: {requirements_text}",
            stream=True,
        )

# =============================
# UI Utilities
# =============================

def render_mermaid(mermaid_code: str, theme: str = "default"):
    """Attempt to render Mermaid via CDN. Falls back to code block if blocked."""
    try:
        import streamlit.components.v1 as components
        html = f"""
        <div class="mermaid">{mermaid_code}</div>
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <script>mermaid.initialize({{ startOnLoad: true, theme: '{theme}' }});</script>
        """
        components.html(html, height=520, scrolling=True)
    except Exception:
        st.code(mermaid_code, language="markdown")


def stream_events_to_chat(stream: Generator[RunContentEvent, None, None], placeholder) -> str:
    """Stream RunContentEvent -> chat-like markdown while accumulating full text."""
    full_text = ""
    for event in stream:
        if event.content:
            full_text += event.content
            placeholder.markdown(full_text)
    return full_text


def fancy_header():
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    col1, col2 = st.columns([0.75, 0.25])
    with col1:
        st.markdown(
            """
            <div class="card">
                <h1>🚀 <span class="gradient">Advanced Requirement Analysis System</span></h1>
                <p>Powered by a coordinated <b>multi‑agent</b> team to turn raw requirements into <i>design‑ready</i> deliverables.</p>
                <div class="badge"><span class="dot"></span> Live Streaming</div>
                <span style="margin:0 .35rem"></span>
                <div class="badge">Groq · Llama‑class models</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.metric("Latency‑friendly", "Streaming", "+ realtime")
        st.metric("Uptime", "99.9%", "+ resilient")


def sidebar_config(default_requirements: str) -> Dict:
    with st.sidebar:
        st.header("⚙️ Configuration")
        groq_api_key = os.getenv("GROQ_API_KEY", "")

        st.divider()
        st.header("📝 Project Details")
        project_name = st.text_input("Project Name", "E‑commerce Platform")

        analysis_type = st.selectbox(
            "Select Analysis Type",
            [
                "Comprehensive",
                "Enterprise",
                "Quick Feature Extraction",
                "User Stories Generation",
                "Technical Architecture Assessment",
            ],
            index=0,
        )

        with st.expander("Advanced toggles"):
            render_mermaid_toggle = st.toggle("Attempt to render Mermaid diagrams")
            mermaid_theme = st.selectbox("Mermaid theme", ["default", "neutral", "forest", "dark"], index=1)
            add_confetti = st.toggle("Celebrate on success 🎉", value=True)

        st.divider()
        st.caption("Need inspiration? Inject a quick sample 👇")
        if st.button("Insert sample requirements", use_container_width=True):
            st.session_state["requirements_text"] = default_requirements
            st.toast("Sample inserted!", icon="✅")

    return {
        "groq_api_key": groq_api_key,
        "project_name": project_name,
        "analysis_type": analysis_type,
        "render_mermaid_toggle": render_mermaid_toggle,
        "mermaid_theme": mermaid_theme,
        "add_confetti": add_confetti,
    }


# =============================
# Main App
# =============================

def main():
    st.set_page_config(
        page_title="Advanced Requirement Analysis System",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://docs.streamlit.io",
            "Report a bug": "https://github.com/",
            "About": "Multi‑agent requirement analysis UI (exciting edition)",
        },
    )

    if "history" not in st.session_state:
        st.session_state["history"] = []  # list of dicts with {ts, project, type, result}
    if "requirements_text" not in st.session_state:
        st.session_state["requirements_text"] = ""

    fancy_header()

    # Sample requirements (same semantics as original)
    sample_requirements = (
        "We need to build a modern e-commerce platform where users can browse products,\n"
        "add items to cart, process payments securely, and track their orders.\n"
        "The system should support multiple payment methods, send email notifications,\n"
        "and provide an admin dashboard for inventory management.\n"
        "We expect high traffic and need the system to be scalable and secure."
    )

    cfg = sidebar_config(sample_requirements)

    st.markdown("### ✍️ Compose your requirements")
    c1, c2 = st.columns([2, 1])
    with c1:
        requirements_text = st.text_area(
            "Enter your requirements…",
            key="requirements_area",
            value=st.session_state.get("requirements_text") or sample_requirements,
            height=260,
            placeholder="Paste raw requirements or high‑level goals…",
        )
    with c2:
        st.markdown("""
        <div class="card">
            <div class="step"><div class="num">1</div><div>Paste or write your raw requirements.</div></div>
            <div class="step"><div class="num">2</div><div>Pick an analysis mode in the sidebar.</div></div>
            <div class="step"><div class="num">3</div><div>Click <b>Analyze</b> and watch the live stream.</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.write("\n")
        st.info("Tip: The ‘Enterprise’ mode adds deeper risk & security lenses.")

    # Action row
    run_cols = st.columns([1, 1, 1, 3])
    with run_cols[0]:
        analyze_clicked = st.button("🔎 Analyze Requirements", type="primary", use_container_width=True)
    with run_cols[1]:
        clear_clicked = st.button("🧹 Clear", use_container_width=True)
    with run_cols[2]:
        show_diagram = st.button("🪄 Generate Flowchart (Mermaid)", use_container_width=True)

    if clear_clicked:
        st.session_state["requirements_text"] = ""
        st.rerun()

    # Tabs for output & history
    tab_results, tab_diagrams, tab_history = st.tabs(["📄 Results", "📈 Diagrams", "🕘 History"])

    # --- Analysis flow ---
    if analyze_clicked:
        if not cfg["groq_api_key"]:
            st.error("Please set your GROQ_API_KEY environment variable to proceed.")
            st.stop()
        if not (requirements_text and requirements_text.strip()):
            st.warning("Please enter the software requirements to analyze.")
            st.stop()

        os.environ["GROQ_API_KEY"] = cfg["groq_api_key"]
        analyzer = RequirementAnalyzer()

        with tab_results:
            st.markdown("#### Live analysis stream")
            chat = st.empty()

            # Choose stream according to analysis type
            if cfg["analysis_type"] in ["Comprehensive", "Enterprise"]:
                stream = analyzer.analyze_requirements(
                    requirements_text=requirements_text,
                    project_name=cfg["project_name"],
                    analysis_level=cfg["analysis_type"],
                )
            elif cfg["analysis_type"] == "Quick Feature Extraction":
                stream = analyzer.quick_feature_extraction(requirements_text)
            elif cfg["analysis_type"] == "User Stories Generation":
                stream = analyzer.generate_user_stories_only(requirements_text)
            else:
                stream = analyzer.assess_technical_architecture(requirements_text)

            start = time.time()
            with st.status(f"Running '{cfg['analysis_type']}' analysis…", expanded=True) as status:
                st.write("Agents syncing context, aligning objectives, and allocating workloads…")
                full_result = stream_events_to_chat(stream, chat)
                elapsed = time.time() - start
                status.update(label="Analysis complete", state="complete")

            # Post-processing
            if cfg.get("add_confetti"):
                st.balloons()
            st.success(f"Done in {elapsed:0.1f}s · {len(full_result):,} chars")

            # Download artifacts
            colA, colB = st.columns(2)
            with colA:
                st.download_button(
                    "⬇️ Download Markdown Report",
                    data=full_result.encode("utf-8"),
                    file_name=f"{cfg['project_name'].strip().replace(' ', '_').lower()}_analysis.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with colB:
                st.download_button(
                    "⬇️ Download Raw Text",
                    data=full_result.encode("utf-8"),
                    file_name=f"{cfg['project_name'].strip().replace(' ', '_').lower()}_analysis.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            # Save to history
            st.session_state["history"].insert(
                0,
                {
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "project": cfg["project_name"],
                    "type": cfg["analysis_type"],
                    "result": full_result,
                },
            )

    # --- Diagramming ---
    with tab_diagrams:
        st.markdown("#### Mermaid flow diagrams")
        flow_col1, flow_col2 = st.columns([1, 2])
        with flow_col1:
            flow_type = st.selectbox("Diagram type", ["flowchart", "sequence"], index=0)
            theme_pick = cfg.get("mermaid_theme", "neutral")
            st.caption("Tip: Use ‘sequence’ for request/response flows.")
        with flow_col2:
            pass

        if show_diagram:
            tool = DiagramGenerationTools()
            code = tool.generate_mermaid_diagram(requirements_text or sample_requirements, flow_type)
            st.markdown("**Generated Mermaid code**")
            st.code(code, language="markdown")
            if cfg.get("render_mermaid_toggle"):
                st.markdown("**Rendered preview**")
                render_mermaid(code, theme=theme_pick)
            else:
                st.info("Rendering is disabled. Enable it from the sidebar.")

    # --- History ---
    with tab_history:
        if not st.session_state["history"]:
            st.info("No runs yet. Your analysis history will appear here.")
        else:
            for item in st.session_state["history"]:
                with st.expander(f"{item['ts']} · {item['project']} · {item['type']}"):
                    st.markdown(item["result"])


if __name__ == "__main__":
    main()
