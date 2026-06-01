# BA Assistant — AI Copilot for Requirements Analysis

> **5 AI agents. 60 seconds. Raw input → delivery-ready specs, user stories, risks & diagrams.**

[![Live App](https://img.shields.io/badge/Try%20It-Live%20App-blue?style=flat&logo=streamlit)](https://businessanalysttools.streamlit.app)
[![Website](https://img.shields.io/badge/Website-touseefshaik.com-2563eb?style=flat)](https://touseefshaik.com/tools/ba-assistant)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat&logo=streamlit)](https://streamlit.io)

---

## What It Does

BA Assistant converts messy, unstructured input into a **complete requirements report** using 5 specialized AI agents working in parallel.

**Input** → a raw text description, uploaded document, or pasted BRD  
**Output** → structured markdown report with specs, stories, architecture, risks & Mermaid diagrams

### Real Example

<details>
<summary><b>Input</b> (what you paste in)</summary>

> We need a loan onboarding portal. Customers should apply online with KYC verification, upload documents, and track application status. Admin approval workflow with notifications at each stage.

</details>

<details>
<summary><b>Output</b> (what the 5 agents generate)</summary>

- ✅ **12 functional requirements** — categorized (Onboarding, KYC, Documents, Workflow, Notifications)
- ✅ **5 non-functional requirements** — security, scalability, performance, audit, compliance
- ✅ **8 user stories** with acceptance criteria (e.g. "As a customer, I want to upload my PAN card so my KYC is verified")
- ✅ **MoSCoW-prioritized feature list**
- ✅ **Architecture recommendation** — microservices + document DB + message queue pattern
- ✅ **7 risks identified** with probability, impact, and mitigation strategies
- ✅ **3 Mermaid diagrams** — flow diagram, sequence diagram, state diagram
- ✅ **Export as PDF** or copy-paste into Confluence / Notion / PRD

</details>

---

## Features (v2)

| Feature | Description |
|---------|-------------|
| 🧠 **5-Agent Agno Team** | Requirements Analyst, Backlog Manager, Solution Architect, Risk Assessor, Diagram Generator — all run in parallel |
| 📄 **Document Upload** | Upload images (whiteboard, handwritten notes) or PDFs — Gemini 3.5 Flash extracts text |
| 💬 **Interactive Q&A** | After analysis, ask follow-up questions — the agents refine the output |
| 📋 **4 Fintech Templates** | Loan Onboarding, KYC Portal, Payment Gateway, Insurance Claims — one-click analysis |
| 📊 **PDF Export** | One-click export to professional PDF with fpdf2 |
| ⚡ **Dual Mode** | Fast (single agent, 30s) or Deep (5-agent Team, 60-90s) |
| 🔍 **Mermaid Diagrams** | Flow, sequence, state, class, and ER diagrams rendered inline |

---

## Architecture

```
User Input (text / image / PDF)
        │
        ▼
┌─────────────────────────────┐
│   Agno Team (Enterprise)    │
│                             │
│  ┌─────────┐  ┌──────────┐  │
│  │ Req     │  │ Backlog  │  │
│  │ Analyst │  │ Manager  │  │
│  │ (V4 F)  │  │ (V4 F)   │  │
│  └────┬────┘  └────┬─────┘  │
│       │            │         │
│  ┌────┴────────────┴─────┐  │
│  │    Coordinator        │  │
│  │    (V4 Pro)           │  │
│  └────┬──────────────┬───┘  │
│       │              │      │
│  ┌────┴────┐  ┌──────┴───┐  │
│  │ Solution│  │ Risk     │  │
│  │ Arch    │  │ Assessor │  │
│  │ (V4 F)  │  │ (V4 F)   │  │
│  └─────────┘  └──────────┘  │
│                             │
│  ┌──────────┐               │
│  │ Diagram  │  ← Gemini     │
│  │ Generator│     3.5 Flash │
│  └──────────┘               │
└─────────────────────────────┘
        │
        ▼
  Structured Report (Markdown)
```

### Model Split
- **Coordinator:** DeepSeek V4 Pro — task decomposition & synthesis
- **4 Worker Agents:** DeepSeek V4 Flash — fast parallel analysis
- **Vision (Document Upload):** Gemini 3.5 Flash — image/PDF text extraction

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | [Streamlit](https://streamlit.io) |
| Agent Framework | [Agno](https://agno.ai) (Team mode) |
| LLMs | DeepSeek V4 Pro, DeepSeek V4 Flash, Gemini 3.5 Flash |
| PDF Export | [fpdf2](https://pyfpdf.github.io/fpdf2/) |
| Diagrams | Mermaid.js (rendered inline) |
| Deployment | Streamlit Cloud |

---

## Quick Start

```bash
# Clone
git clone https://github.com/Touseef1949/BA_Assistant.git
cd BA_Assistant

# Install
pip install -r requirements.txt

# Set API keys
export DEEPSEEK_API_KEY="your-key"
export GEMINI_API_KEY="your-key"

# Run
streamlit run app.py
```

---

## Pricing Tiers

| Tier | Price | What You Get |
|------|-------|-------------|
| **Free** | ₹0 | Single-agent analysis (30s), text input only, limited exports |
| **Pro** | ₹499/mo | 5-agent Team, document upload, PDF export, interactive Q&A |
| **Team** | ₹1,999/mo | Everything in Pro + shared workspace, 5 team members |

---

## Roadmap

- [ ] Confluence / Jira export integration
- [ ] Custom BRD templates (upload your company's format)
- [ ] Team collaboration (shared analyses, comments)
- [ ] Batch analysis (process multiple BRDs at once)
- [ ] API access for CI/CD pipelines

---

*Built by [Touseef Shaik](https://touseefshaik.com) — AI Product Owner at Broadridge. Previously 12+ years at CGI building AI products for enterprise. Now building AI tools for BAs and Product Owners.*
