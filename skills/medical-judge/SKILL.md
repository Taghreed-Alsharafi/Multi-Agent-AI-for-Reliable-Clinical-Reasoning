---
name: medical-judge
description: "Evaluates and synthesizes reviews from multiple medical specialists, resolves conflicts, and produces a final consolidated document. Use for: consolidating multi-specialty medical reviews, ensuring consistency, and generating a final comprehensive report."
---

# Medical Judge

This skill is responsible for the final stage of the multi-agent medical document review workflow. It takes the individual reviews from various medical specialists, identifies overlaps and conflicts, resolves them, and synthesizes a single, coherent, and comprehensive final medical report.

## Workflow

1.  **Receive Specialist Reviews**: Collect all individual reviews provided by the `medical-specialist` agents, along with the original medical document and the designated lead specialist.
2.  **Identify Overlaps and Conflicts**: Analyze the key findings and recommendations from each specialist. Look for redundant information, contradictory statements, or areas where different specialists have focused on the same aspect with varying conclusions.
3.  **Resolve Conflicts**: Prioritize information based on clinical significance, the expertise of the contributing specialist, and the guidance from the designated Lead Specialist. If direct conflicts exist, attempt to reconcile them into a unified statement or highlight the differing opinions with justification.
4.  **Synthesize Final Document**: Combine all relevant and reconciled information into a single, comprehensive medical report. Ensure a logical flow and clear presentation.
5.  **Ensure Consistency and Quality**: Review the consolidated report for consistency in terminology, formatting, and overall clinical accuracy and completeness.

## Input

*   **`original_document`**: The initial medical document that was reviewed.
*   **`specialist_reviews`**: A list of structured reviews (e.g., Markdown strings) from each `medical-specialist` agent.
*   **`lead_specialist`**: The name of the specialist designated as the lead by the `medical-supervisor` skill.

## Output

Provide a single, consolidated medical report in Markdown format. The report should include:

*   **Overall Summary**: A brief overview of the case and the key conclusions.
*   **Key Findings by Specialty**: A section summarizing the most important findings from each contributing specialist.
*   **Consolidated Recommendations**: A unified list of all actionable recommendations, prioritized and reconciled.

## Example Output Format

```markdown
# Consolidated Medical Report

## Overall Summary
Patient presented with acute chest pain, elevated troponin, and new-onset atrial fibrillation. Initial assessment indicated a primary cardiac event with secondary renal involvement. Multiple specialists reviewed the case to ensure comprehensive care.

## Key Findings by Specialty

### Cardiologist (Lead Specialist)
*   New-onset atrial fibrillation with rapid ventricular response.
*   Elevated troponin suggesting acute coronary syndrome.

### Nephrologist
*   Acute Kidney Injury (AKI) with creatinine increase from 1.1 to 2.4.
*   Electrolyte imbalances noted.

### Pharmacist
*   Identified potential drug-drug interactions with current medication regimen.
*   Recommended dose adjustments for renal impairment.

## Consolidated Recommendations

1.  **Cardiac Management**: Initiate rate control with beta-blocker and consider anticoagulation. Follow-up with outpatient cardiology.
2.  **Renal Management**: Monitor renal function closely, adjust medications for AKI, and manage electrolyte imbalances.
3.  **Medication Review**: Implement pharmacist-recommended medication adjustments and reconcile drug list.
4.  **Overall Coordination**: The Cardiologist will oversee the integrated care plan and ensure communication between specialties.
```
