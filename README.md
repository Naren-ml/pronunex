<div align="center">
  <img src="frontend/public/icon.png" alt="Pronunex Logo" width="100">
</div>

# Pronunex

**Intelligent Speech Therapy Platform with Adaptive Exercises and Progress Tracking**

An AI-powered speech therapy system that evaluates pronunciation accuracy, identifies phoneme-level errors, and provides personalized practice exercises to improve spoken English.

## Technology Stack

### Backend
![Django](https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=django&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)
![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)

Django REST Framework • Wav2Vec2 • librosa • PyTorch • Transformers

### Frontend
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)

React Router • Recharts • Lucide Icons

### AI & LLM
![Google Gemini](https://img.shields.io/badge/Google_Gemini-8E75B2?style=for-the-badge&logo=google&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-000000?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBkPSJNMTIgMkM2LjQ4IDIgMiA2LjQ4IDIgMTJzNC40OCAxMCAxMCAxMCAxMC00LjQ4IDEwLTEwUzE3LjUyIDIgMTIgMnoiIGZpbGw9IiNmZmYiLz48L3N2Zz4=&logoColor=white)

Google Gemini • Groq • Cerebras

## Key Features

- **Pronunciation Assessment**: Phoneme-level analysis using Wav2Vec2 embeddings and cosine similarity scoring
- **Adaptive Practice**: AI-generated sentences targeting weak phonemes
- **Progress Tracking**: Visual dashboards showing improvement over time
- **Personalized Feedback**: Human-readable explanations powered by LLMs
- **Reference Library**: 44 English phonemes with articulation guidance

## Architecture Principles

This project follows strict architectural guidelines to maintain scientific validity:

1. **Separation of Concerns**: NLP Core handles signal processing; LLM Engine only generates feedback text
2. **Deterministic Scoring**: All pronunciation scores use cosine similarity on embeddings
3. **LLM Constraints**: LLMs never score pronunciation, detect phonemes, or judge speech correctness
4. **Precomputed Data**: Reference phonemes and embeddings are cached, not regenerated per request

## Project Structure

```
pronunex/
├── backend/                 # Django backend
│   ├── apps/               # Django applications
│   │   ├── accounts/       # User authentication
│   │   ├── library/        # Phonemes & reference sentences
│   │   ├── practice/       # Sessions, attempts, assessment
│   │   ├── analytics/      # Progress tracking
│   │   ├── llm_engine/     # LLM feedback generation
│   │   └── sentence_engine/ # Adaptive sentence generation
│   ├── nlp_core/           # Signal processing modules
│   ├── services/           # Shared services
│   ├── config/             # Django configuration
│   └── media/              # Audio storage (gitignored)
│
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # Reusable UI components
│   │   ├── pages/          # Page components
│   │   ├── context/        # React context providers
│   │   ├── hooks/          # Custom React hooks
│   │   ├── api/            # API client
│   │   └── styles/         # Global styles
│   └── public/             # Static assets
│
└── docs/                   # Documentation
```

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- PostgreSQL (or use Supabase)

### Installation

```bash
# Clone repository
git clone https://github.com/infosys-springboard-vinternship/pronunex.git
cd pronunex

# Backend setup
cd backend
python -m venv venv
venv\Scripts\activate  # Windows | source venv/bin/activate (Linux/Mac)
pip install -r requirements.txt
cp .env.example .env  # Configure your API keys
python manage.py migrate
python manage.py seed_data
python manage.py runserver  # http://localhost:8000

# Frontend setup (new terminal)
cd frontend
npm install
cp .env.example .env
npm run dev  # http://localhost:5173
```

For detailed setup instructions, see:
- [Backend README](backend/README.md) - Complete backend setup, API documentation, and architecture
- [Frontend README](frontend/README.md) - Frontend setup, component structure, and development guide

## API Overview

RESTful APIs available at `http://localhost:8000/api/v1/`:

- `/auth/` - Authentication and user management
- `/library/` - Phonemes and reference sentences
- `/practice/` - Practice sessions and assessments
- `/analytics/` - Progress tracking and statistics

**Admin Panel**: `http://localhost:8000/admin/`

See [Backend README](backend/README.md) for complete API documentation.


## Development

See detailed development guides:
- [Backend README](backend/README.md) - Django commands, testing, deployment
- [Frontend README](frontend/README.md) - npm scripts, component development, styling


## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and development process.

### Key Guidelines

- Follow the architecture principles outlined above
- Never use LLMs for pronunciation scoring
- Maintain separation between NLP Core and LLM Engine
- Write tests for new features
- Update documentation as needed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Academic Context

This project was developed as part of the Infosys Springboard Virtual Internship program. It demonstrates the application of AI and NLP techniques to speech therapy, with a focus on:

- Production-grade architecture
- Scientific validity and reproducibility
- Clear separation of AI responsibilities
- Measurable user outcomes

## Acknowledgments

- Infosys Springboard for the internship opportunity
- Wav2Vec2 team for the alignment model
- Open-source community for the tools and libraries used

## Support

For issues and questions, please use the GitHub issue tracker.

---

**Repository**: https://github.com/infosys-springboard-vinternship/pronunex.git
