# EstateAssess STAR 🛡️

A premium, AI-powered recruitment assessment platform designed specifically for high-stakes real estate candidate evaluation using the **STAR Methodology** (Skill, Training, Attitude, Results).

![v3.4](https://img.shields.io/badge/TailwindCSS-v3.4-38B2AC)
![Vite](https://img.shields.io/badge/Vite-7.3-646CFF)
![Gemini](https://img.shields.io/badge/AI-Gemini_3.0_Flash-88ddff)

## 🌟 Key Features

-   **STAR Assessment Engine**: A mixed-modality interview flow supporting both numeric ratings and descriptive MCQs.
-   **Gemini 3.0 AI Generation**: Live, mid-interview question generation using context-aware reasoning for deep candidate insights.
-   **Kaggle HR Bank**: Built-in repository of 50+ real-world HR questions sourced from high-quality professional datasets.
-   **Visual Analytics**: Real-time Radar (Spider) chart integration to visualize candidate competency across all four STAR pillars.
-   **Premium UI**: Glassmorphism design system built with Navy, Gold, and Slate accents for a state-of-the-art panelist experience.

## 🛠️ Technology Stack

-   **Frontend**: React 19 + Vite
-   **Styling**: Tailwind CSS v3.4 + shadcn/ui + Framer Motion
-   **Data Science**: Python (KaggleHub) + Pandas
-   **AI**: Google Generative AI SDK (Gemini 3.0 Flash Preview)
-   **Charts**: Recharts

## 🚀 Getting Started

### Prerequisites

-   Node.js (LTS)
-   Google AI Studio API Key ([Get one here](https://aistudio.google.com/))

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/mathurshubham/real-estate-talent-screening.git
    cd real-estate-talent-screening
    ```

2.  **Install dependencies**:
    ```bash
    npm install
    ```

3.  **Environment Setup**:
    Create a `.env` file in the root directory:
    ```env
    VITE_GEMINI_API_KEY=your_actual_key_here
    ```

4.  **Run Development Server**:
    ```bash
    npm run dev
    ```

## 📁 Project Structure

-   `/docs/mvp/`: Contains the v4.0 implementation plan, walkthrough, and task logs.
-   `/src/components/`: Reusable UI components.
-   `/src/data/`: STAR pillar definitions and Kaggle dataset.
-   `fetch_kaggle_data.py`: Python utility for refreshing the HR question bank.

## 📄 Documentation

For a detailed breakdown of the development journey and architectural decisions, see the [Walkthrough](docs/mvp/walkthrough.md).
