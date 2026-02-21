# implementation_plan.md: Advanced STAR Assessment with Kaggle & AI (v4.0)

Goal: Supercharge the recruitment process by leveraging external datasets and AI-driven question generation.

## Proposed Changes

### [Tailwind CSS Fix]
- Resolve the "unstyled" issue by ensuring Tailwind CSS v3.4 is correctly configured with `postcss` and injected into the Vite build.
- Standardize on ESM (`export default`) for all config files.

### [Data Integration: Kaggle]
- **KaggleHub Scraper**: Run a Python utility to fetch 50 HR interview questions from the `aryan208` dataset.
- **Data Pipeline**: Clean and map these questions into our four-pillar format (Skill, Training, Attitude, Results).
- **Storage**: Save as `src/data/kaggleQuestions.json`.

### [AI Feature: Gemini Flash]
- **AI Generator**: Add a "Smart Generate" button that uses a specific prompt to create interview questions.
- **Context Injection**: The prompt will include candidate role, current pillar context, and STAR parameters.
- **Client-side Implementation**: Provide a structure for AI calls (using a mock/placeholder until direct API keys are configured).

### [UI/UX Upgrades]
- **MCQ + Rating Hybrid**: Ensure the question views are consistent for both numeric and text-based choices.
- **Dynamic Grid**: Dashboard will allow selecting between "Standard Library" and "Kaggle Library".

## Verification Plan

### Manual Verification
1. **Style Audit**: Verify professional styling is restored (Navy/Gold buttons, cards).
2. **Kaggle Data**: Confirm that 50 new questions appear in the selection bank.
3. **AI Generation**: Test the "Generate" button to see if it adds a valid question to the session.
4. **Radar Integration**: Ensure new questions contribute correctly to the final spider chart.
