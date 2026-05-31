import streamlit as st
from agno.agent import Agent
from agno.models.groq import Groq
from agno.team.team import Team
from agno.tools.reasoning import ReasoningTools
import os
import dotenv
from typing import Generator

# Import the specific event model to handle the stream output
from agno.agent import RunContentEvent


# Attempt to load environment variables from a .env file
try:
    dotenv.load_dotenv()
except Exception as e:
    print(f"Error loading .env file (this is expected if running in a deployed environment): {e}")


# --- Original Agent and Team Definitions (No Changes Needed Here) ---

# Custom tool for diagram generation (Implementation added)
class DiagramGenerationTools:
    def generate_mermaid_diagram(self, requirements: str, flow_type: str = "flowchart") -> str:
        """
        Generate Mermaid diagram code based on requirements.
        Note: This is a conceptual implementation. A real implementation would involve
        a language model call to transform requirements into Mermaid syntax.
        """
        # A simple, rule-based example for a flowchart.
        # A more advanced version would use an LLM to generate the diagram.
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
            # Placeholder for a sequence diagram
            return "sequenceDiagram\n    participant User\n    participant System\n    User->>System: Browse products\n    System-->>User: Display products"
        else:
            return f"graph TD\n    A[Diagram type '{flow_type}' not supported in this example.]"


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

# --- RequirementAnalyzer Class (No Changes Needed Here) ---
class RequirementAnalyzer:
    """ A class to analyze software requirements using a team of AI agents. """
    def __init__(self):
        self.team = requirement_analysis_team

    def analyze_requirements(self, requirements_text: str, project_name: str, analysis_level: str) -> Generator[RunContentEvent, None, None]:
        """ Main function to analyze requirements using the agent team. """
        analysis_instructions = {
            "Comprehensive": "Provide detailed analysis with full feature breakdown, user stories, and technical assessment",
            "Enterprise": "Provide enterprise-level analysis including detailed risk assessment, security considerations, and scalability planning"
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
        """Quick feature extraction using just the feature extraction agent."""
        return feature_extraction_agent.run(
            f"Extract and prioritize features from these requirements: {requirements_text}",
            stream=True
        )

    def generate_user_stories_only(self, requirements_text: str) -> Generator[RunContentEvent, None, None]:
        """Generate user stories using just the user story agent."""
        return user_story_agent.run(
            f"Create user stories with acceptance criteria for: {requirements_text}",
            stream=True
        )

    def assess_technical_architecture(self, requirements_text: str) -> Generator[RunContentEvent, None, None]:
        """Get technical recommendations using the architecture agent."""
        return technical_architecture_agent.run(
            f"Provide technical architecture recommendations for: {requirements_text}",
            stream=True
        )


# --- HELPER FUNCTION (No Changes Needed Here) ---
def get_content_stream(stream: Generator[RunContentEvent, None, None]) -> Generator[str, None, None]:
    """
    Takes a generator of RunContentEvent objects and yields the content string from each.
    This adapts the agno stream for st.write_stream.
    """
    for event in stream:
        if event.content:
            yield event.content


# --- STREAMLIT UI (No Changes Needed Here) ---
def main():
    st.set_page_config(
        page_title="Advanced Requirement Analysis System",
        page_icon="🚀",
        layout="wide"
    )

    st.title("🚀 Advanced Requirement Analysis System")
    st.markdown("Powered by a multi-agent team to deliver comprehensive software development insights.")

    # --- SIDEBAR FOR CONFIGURATION ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        groq_api_key = st.text_input(
            "GROQ API Key", 
            type="password", 
            help="Get your key from [Groq Console](https://console.groq.com/keys)",
            value=os.getenv("GROQ_API_KEY", "")
        )
        
        st.header("📝 Project Details")
        project_name = st.text_input("Project Name", "E-commerce Platform")

        analysis_type = st.selectbox(
            "Select Analysis Type",
            [
                "Comprehensive", 
                "Enterprise",
                "Quick Feature Extraction",
                "User Stories Generation",
                "Technical Architecture Assessment"
            ],
            index=0
        )
        
        st.markdown("---")
        st.info("Provide your software requirements in the main text area and click 'Analyze Requirements' to begin.")

    # --- MAIN CONTENT AREA ---
    st.header("📋 Enter Your Software Requirements")

    sample_requirements = """We need to build a modern e-commerce platform where users can browse products, 
add items to cart, process payments securely, and track their orders. 
The system should support multiple payment methods, send email notifications, 
and provide an admin dashboard for inventory management. 
We expect high traffic and need the system to be scalable and secure."""

    requirements_text = st.text_area(
        "Enter your requirements here...",
        value=sample_requirements,
        height=300
    )

    if st.button("Analyze Requirements", type="primary"):
        if not groq_api_key:
            st.error("Please enter your GROQ API Key in the sidebar to proceed.")
            st.stop()
        
        if not requirements_text.strip():
            st.warning("Please enter the software requirements to analyze.")
            st.stop()
            
        os.environ["GROQ_API_KEY"] = groq_api_key
        
        analyzer = RequirementAnalyzer()

        st.header("📄 Analysis Results")
        
        try:
            with st.spinner(f"Running '{analysis_type}' analysis... The agent team is on it!"):
                response_container = st.empty()
                
                stream_of_events = None
                # Select the correct analysis method based on user choice
                if analysis_type in ["Comprehensive", "Enterprise"]:
                    stream_of_events = analyzer.analyze_requirements(
                        requirements_text=requirements_text,
                        project_name=project_name,
                        analysis_level=analysis_type
                    )
                elif analysis_type == "Quick Feature Extraction":
                    stream_of_events = analyzer.quick_feature_extraction(requirements_text)
                elif analysis_type == "User Stories Generation":
                    stream_of_events = analyzer.generate_user_stories_only(requirements_text)
                elif analysis_type == "Technical Architecture Assessment":
                    stream_of_events = analyzer.assess_technical_architecture(requirements_text)
                
                if stream_of_events:
                    # Wrap the event stream with our helper function to extract content
                    content_stream = get_content_stream(stream_of_events)
                    # Use st.write_stream to display the clean string output
                    response_container.write_stream(content_stream)

            st.success("Analysis complete!")

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            st.exception(e)

if __name__ == "__main__":
    main()