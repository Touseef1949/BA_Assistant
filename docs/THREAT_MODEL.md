# AI threat model

## Protected assets

- uploaded and pasted requirements, generated reports, and report history
- account identifiers, authentication sessions, usage quotas, and payment state
- DeepSeek, Gemini, Supabase, and Razorpay credentials
- system prompts, model configuration, and operational logs

## Trust boundaries

Untrusted user text and files cross into local parsers and model prompts.
Model output then crosses back into rendered Markdown, diagrams, and downloads.
Authentication, payment providers, model providers, and the hosting platform
are external systems and must not be treated as trusted storage.

## Principal threats and controls

| Threat | Control |
| --- | --- |
| Prompt injection in requirements or uploaded documents | Treat input as data, retain fixed system instructions, and never grant model output tool or secret access. |
| Secret exposure in source, logs, or model prompts | Deployment secrets, `.gitignore`, push protection, Gitleaks, redacted errors, and no secret values in prompts. |
| Cross-user report disclosure | Verified identity, salted hashed history filenames, and per-user lookup boundaries. |
| Unsafe generated requirements or diagrams | Human review is required; exports are drafts and Mermaid is sanitized before rendering. |
| Cost or quota abuse | Authentication, usage limits, explicit paid-mode gates, and health/load checks that never invoke models. |
| Malicious or oversized files | Constrained upload types, local PDF extraction, explicit image extraction, and bounded processing. |
| Dependency or workflow compromise | Locked dependencies, Dependabot, least-privilege workflow tokens, and immutable action pins. |

## Privacy boundary

Do not submit secrets, production credentials, regulated personal data, or
customer material that is not approved for the configured model providers.
Local PDF parsing does not make the full workflow local: extracted text can be
sent to the configured analysis model when a report is generated.

## Residual risk

Model output can be incomplete, incorrect, or influenced by adversarial input.
BA Assistant supports analysis; it does not replace stakeholder validation,
security review, legal review, or delivery approval.
