# ImageLynx Project: Conversation Transfer Context

This document summarizes the progress, context, and generated assets from the current AI chat session. It is designed to be provided as initial context to a new AI model to seamlessly continue the work.

## 1. Project Context
- **Project**: ImageLynx
- **Domain**: Computational physiology and haemodynamics modelling of the carotid body microvasculature.
- **Core Pipeline**: Extracts 3D vascular networks from microscopy images (e.g., using Ilastik probabilities and Shannon entropy filtering) and solves for steady-state blood flow, spatially varying rheology (Pries-Secomb), and tissue oxygen perfusion (coupled 1D-3D advection-diffusion-reaction).

## 2. Completed Work & Milestones
We executed a rigorous, two-phase process to audit and upgrade the project's conceptual documentation against the Python source code.

### Phase A: Audit
- Used subagents to extract exhaustive inventories (functions, equations, constants, guard rails, solver details) from core source files (`poiseuille.py`, `resistance.py`, `rheology.py`, `perfusion.py`, `carotid_image_to_model.py`).
- Cross-referenced the inventories against the original `cb_image_to_model_modelling_capabilities_conceptual_summary.md`.
- Generated **`audit_report.md`**, identifying **23 specific deficiencies**:
  - **1 Critical**: Fixed a 12-order-of-magnitude error in the PCO₂ clamp (document said `1e-12`, code actually used `1.0`).
  - **14 Medium**: Missing function attributions, undocumented guard rails, missing constants, and an incomplete flux equation.
  - **8 Low**: Minor constant omissions, notation inconsistencies.

### Phase B: Targeted Upgrade
- Created a new, upgraded document: **`cb_image_to_model_modelling_capabilities_conceptual_summary_v2.md`**.
- Applied all 23 surgical fixes to the `v2` document while preserving the original structure and correct content.
- Generated a **`walkthrough.md`** summarizing the specific changes made.

### Process Reverse-Engineering
- Distilled the successful Phase A/Phase B methodology into a highly robust, reusable prompt template: **`master_prompt_template.md`**. This template allows other models to perform similar source-code-grounded documentation audits.

## 3. Key Technical Discussions
- **Shannon Entropy Filtering**: We reviewed how the pipeline handles 4D Ilastik probability outputs. The pipeline uses Shannon entropy as a *quality filter*: voxels with normalized entropy > 0.95 (high uncertainty) have their vessel probabilities zeroed out before standard probability-based thresholding (Hysteresis/Otsu) is applied.

## 4. Pending / Next Task
Right before the chat transfer, the user requested the following:
> "Create an LLM optimised prompt for the following request: Add the following content to the v2 conceptual summary: For all the modelling and computational capabilities presented within the document, add content regarding how each of these capabilities are verification tested within the pipeline (i.e. what are the associated testing/unit testing methods present that verify the outputs of the model are behaving as expected)."

**Notes for the next AI:**
- A quick scan of the `/home/dsas627/PycharmProjects/ImageLynx/tests/` directory revealed 36 test files (e.g., `test_haemodynamics_analytical.py`, `test_haemodynamics_perfusion.py`, `test_synthetic_network_statistics.py`).
- The next step should be to ask the user clarifying questions (if any) to formulate a robust prompt that directs an LLM to map the concepts in the `v2` summary to these specific testing methodologies.

## 5. Artifact Directory Reference
The following key files were generated and are available in the workspace/artifact directories:
- `examples/cb_image_to_model_modelling_capabilities_conceptual_summary_v2.md` (The upgraded target document)
- `audit_report.md` (The 23-point finding list)
- `master_prompt_template.md` (The reusable audit prompt)
- `walkthrough.md` (Summary of applied fixes)
