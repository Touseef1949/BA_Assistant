from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.models.groq import Groq
from agno.models.openai import OpenAIChat
from agno.team.team import Team
from agno.tools.reasoning import ReasoningTools
import os, dotenv

dotenv.load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# from agno.tools.web_search import WebSearchTools
from typing import Dict, List, Any
import json
from datetime import datetime

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
    # add_datetime_to_instructions=True,
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
    # add_datetime_to_instructions=True,
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
    # add_datetime_to_instructions=True,
)

# Agent 4: Technical Architecture Agent
technical_architecture_agent = Agent(
    name="Technical Architecture Agent",
    role="Senior software architect specializing in technology recommendations and system design",
    model=Groq(id="openai/gpt-oss-120b"),
    # tools=[WebSearchTools()],
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
    # add_datetime_to_instructions=True,
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
    # add_datetime_to_instructions=True,
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
    # add_datetime_to_instructions=True,
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
    # add_datetime_to_instructions=True,
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
    # add_datetime_to_instructions=True,
)

# Main Requirement Analysis Team
requirement_analysis_team = Team(
    name="Advanced Requirement Analysis Team",
    # mode="coordinate",  # Agents work together in coordination
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
    show_members_responses=False,  # Set to True for debugging
    # enable_agentic_context=True,
    # add_datetime_to_instructions=True,
#     success_criteria="""The team has provided a complete requirement analysis including:
#         1. Structured requirement parsing with identified entities and business rules
#         2. Prioritized feature list with dependencies and complexity assessment
#         3. Complete user stories with acceptance criteria and story point estimates
#         4. Technical architecture recommendations with justified technology choices
#         5. Detailed process flow diagrams in Mermaid format
#         6. Comprehensive risk assessment with mitigation strategies
#         7. Quality validation ensuring consistency across all deliverables
#         8. Executive summary with project overview and implementation roadmap
#         All outputs must be professionally structured, consistent, and actionable."""
)

# Specialized Analysis Functions
class RequirementAnalyzer:
    def __init__(self):
        self.team = requirement_analysis_team
        self.analysis_history = []

    def analyze_requirements(self, requirements_text: str, project_name: str = None, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Main function to analyze requirements using the agent team
        
        Args:
            requirements_text: Raw requirement text from user
            project_name: Optional project identifier
            analysis_type: Type of analysis (quick, comprehensive, enterprise)
        
        Returns:
            Structured analysis results
        """
        
        # Customize analysis based on type
        analysis_instructions = {
            "quick": "Provide a rapid analysis focusing on core features and basic user stories",
            "comprehensive": "Provide detailed analysis with full feature breakdown, user stories, and technical assessment",
            "enterprise": "Provide enterprise-level analysis including detailed risk assessment, security considerations, and scalability planning"
        }
        
        prompt = f"""
        Analyze the following software requirements for {project_name or 'this project'}:
        
        REQUIREMENTS:
        {requirements_text}
        
        ANALYSIS TYPE: {analysis_type}
        SPECIAL INSTRUCTIONS: {analysis_instructions.get(analysis_type, '')}
        
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
        
        # Execute team analysis
        response = self.team.print_response(
            prompt,
            stream=True,
            show_full_reasoning=True,
            stream_intermediate_steps=True,
        )
        
        # Store analysis in history
        analysis_record = {
            "timestamp": datetime.now().isoformat(),
            "project_name": project_name,
            "requirements": requirements_text,
            "analysis_type": analysis_type,
            "results": response
        }
        self.analysis_history.append(analysis_record)
        
        return analysis_record

    def quick_feature_extraction(self, requirements_text: str) -> Dict[str, Any]:
        """Quick feature extraction using just the feature extraction agent"""
        return feature_extraction_agent.print_response(
            f"Extract and prioritize features from these requirements: {requirements_text}",
            stream=True
        )

    def generate_user_stories_only(self, requirements_text: str) -> Dict[str, Any]:
        """Generate user stories using just the user story agent"""
        return user_story_agent.print_response(
            f"Create user stories with acceptance criteria for: {requirements_text}",
            stream=True
        )

    def assess_technical_architecture(self, requirements_text: str) -> Dict[str, Any]:
        """Get technical recommendations using the architecture agent"""
        return technical_architecture_agent.print_response(
            f"Provide technical architecture recommendations for: {requirements_text}",
            stream=True
        )

# Usage Examples
if __name__ == "__main__":
    # Initialize the analyzer
    analyzer = RequirementAnalyzer()

    # Example 1: Comprehensive Analysis
    sample_requirements = """
    We need to build a modern e-commerce platform where users can browse products, 
    add items to cart, process payments securely, and track their orders. 
    The system should support multiple payment methods, send email notifications, 
    and provide an admin dashboard for inventory management. 
    We expect high traffic and need the system to be scalable and secure.
    """

    print("🚀 Starting Comprehensive Requirement Analysis...")
    print("=" * 60)

    # Run comprehensive analysis
    results = analyzer.analyze_requirements(
        requirements_text=sample_requirements,
        project_name="E-commerce Platform",
        analysis_type="comprehensive"
    )

    print("\n" + "=" * 60)
    print("✅ Analysis Complete!")

    # Example 2: Quick Feature Extraction
    print("\n🔍 Running Quick Feature Extraction...")
    print("-" * 40)

    quick_features = analyzer.quick_feature_extraction(sample_requirements)

    # Example 3: User Stories Generation
    print("\n📝 Generating User Stories...")
    print("-" * 40)

    user_stories = analyzer.generate_user_stories_only(sample_requirements)

    # Example 4: Technical Architecture Assessment
    print("\n🏗️ Technical Architecture Assessment...")
    print("-" * 40)

    tech_assessment = analyzer.assess_technical_architecture(sample_requirements)

    print("\n" + "=" * 60)
    print("🎉 All analyses completed successfully!")
    print(f"Total analyses in history: {len(analyzer.analysis_history)}")