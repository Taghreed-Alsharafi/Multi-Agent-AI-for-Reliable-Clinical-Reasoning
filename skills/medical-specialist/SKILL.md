---
name: medical-specialist
description: "Provides a framework for a medical professional to review and revise documents based on a specific clinical specialty. Use for: detailed review of medical documents, identifying specialty-specific findings, and providing expert feedback."
---

# Medical Specialist

This skill defines the general workflow for a medical professional to review and revise medical documents from the perspective of a specific clinical specialty. It is designed to be adaptable to various roles (e.g., Cardiologist, Neurologist, Radiologist) by incorporating role-specific guidance.

## Workflow

1.  **Understand the Role**: The skill will be invoked with a specific medical role (e.g., "Cardiologist").
2.  **Load Role-Specific Guidance**: Read `/home/ubuntu/skills/medical-specialist/references/role-guidance.md` for general instructions relevant to all specialists, and potentially load a more specific guidance file if available (e.g., `cardiologist-guidance.md`).
3.  **Review Document**: Analyze the provided medical document, focusing on aspects relevant to the assigned specialty.
4.  **Identify Key Findings**: Extract and summarize critical information, diagnoses, treatments, or concerns pertinent to the specialty.
5.  **Propose Revisions/Feedback**: Based on the review, suggest specific revisions, provide expert opinions, or highlight areas requiring further attention.

## Input

*   **`medical_document`**: The full text of the medical document to be reviewed.
*   **`specialty_role`**: The specific medical role assigned to this specialist (e.g., "Cardiologist", "Radiologist").

## Output

Provide a structured review in Markdown format, including:

*   **Specialty**: The assigned role.
*   **Key Findings**: Bulleted list of relevant observations.
*   **Recommendations/Revisions**: Bulleted list of suggested changes or expert opinions.

## Example Output Format

```markdown
### Cardiologist Review

**Key Findings:**
*   Patient presents with new-onset atrial fibrillation with rapid ventricular response.
*   Echocardiogram shows mild left ventricular hypertrophy.

**Recommendations/Revisions:**
*   Initiate rate control with beta-blocker (e.g., Metoprolol 25mg BID).
*   Consider anticoagulation with Apixaban given CHA2DS2-VASc score of 2.
*   Recommend follow-up with outpatient cardiology within 2 weeks.
```

## Resources

### references/

*   **`role-guidance.md`**: General guidelines for all medical specialists. More specific guidance files (e.g., `cardiologist-guidance.md`) can be added here for detailed instructions per specialty.
