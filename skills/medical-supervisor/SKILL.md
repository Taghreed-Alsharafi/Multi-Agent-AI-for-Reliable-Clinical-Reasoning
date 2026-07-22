---
name: medical-supervisor
description: "Analyzes medical documents to determine the required clinical specialists for review and revision. Use for: identifying medical specialties, determining the number of reviewers, and coordinating multi-agent medical workflows."
---

# Medical Supervisor

This skill acts as the "first responder" in a multi-agent medical document review workflow. It analyzes the content of a medical document (e.g., discharge summary, case report, clinical trial data) to determine which clinical specialists are needed to ensure a comprehensive and accurate review.

## Workflow

1.  **Analyze the Document**: Read the provided medical document thoroughly.
2.  **Identify Key Clinical Domains**: Look for keywords, diagnostic tests, medications, and symptoms that indicate specific medical specialties.
3.  **Consult the Specialist Mapping**: Read `/home/ubuntu/skills/medical-supervisor/references/specialist-mapping.md` to match clinical indicators with the appropriate specialists.
4.  **Determine the Number of Specialists**:
    *   For simple, single-system cases (e.g., uncomplicated pneumonia), select **1-2** specialists (e.g., Pulmonologist, Pharmacist).
    *   For complex, multi-system cases (e.g., septic shock with AKI and heart failure), select **3-5** specialists (e.g., Infectious Disease, Nephrologist, Cardiologist, Pulmonologist, Pharmacist).
    *   Always include a **Pharmacist** if the document involves complex medication regimens or polypharmacy.
    *   Always include a **Radiologist** if the document relies heavily on imaging findings.
5.  **Designate Lead Specialist**: Based on the primary clinical focus of the document and the 'Lead Criteria' in the specialist mapping, identify one specialist as the 'Lead'. The Lead Specialist will be responsible for overall coordination and final review.
6.  **Output the Selection**: Provide a clear list of the selected specialists, indicating the Lead Specialist, and a brief justification for each based on the document's content.

## Example Output Format

| Specialist | Role | Justification |
| :--- | :--- | :--- |
| **Cardiologist** | Lead Specialist | Patient has a history of heart failure and presented with elevated troponin levels, indicating a primary cardiac issue. |
| **Nephrologist** | Reviewer | Creatinine increased from 1.1 to 2.4, indicating Acute Kidney Injury (AKI). |
| **Pharmacist** | Reviewer | Patient is on 12+ medications; requires reconciliation and dosing adjustment for AKI. |

## Next Steps

Once the specialists are identified, the next phase of the workflow is to invoke the specific **Medical Specialist Skills** for each role identified.
