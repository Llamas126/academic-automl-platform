# Guía de Contribución / Contributing Guide

¡Gracias por tu interés en contribuir a **academic-automl-platform**! Este
documento describe cómo configurar tu entorno, desarrollar, probar y enviar
cambios.

Thank you for your interest in contributing to **academic-automl-platform**! This
document describes how to set up your environment, develop, test, and submit
changes.

---

## Tabla de Contenidos / Table of Contents

- [Requisitos Previos / Prerequisites](#requisitos-previos--prerequisites)
- [Clonar el Repositorio / Clone the Repository](#clonar-el-repositorio--clone-the-repository)
- [Configuración / Setup](#configuración--setup)
- [Estructura del Proyecto / Project Structure](#estructura-del-proyecto--project-structure)
- [Desarrollo / Development](#desarrollo--development)
- [Estilo de Código / Code Style](#estilo-de-código--code-style)
- [Tests](#tests)
- [Commits](#commits)
- [Pull Requests](#pull-requests)
- [Reportar Bugs / Report Bugs](#reportar-bugs--report-bugs)
- [Solicitar Funcionalidades / Request Features](#solicitar-funcionalidades--request-features)

---

## Requisitos Previos / Prerequisites

- **Python** >= 3.13
- **Docker** con Compose v2 (para stack completo)
- **Node.js** >= 18 (para el frontend)
- **Git**
- **Redis** (opcional para desarrollo local sin Docker)

**Python** >= 3.13, **Docker** with Compose v2 (for full stack), **Node.js** >= 18
(for frontend), **Git**, **Redis** (optional for local development without Docker).

---

## Clonar el Repositorio / Clone the Repository

```bash
git clone https://github.com/Llamas126/academic-automl-platform.git
cd academic-automl-platform
```

---

## Configuración / Setup

### Backend (Python)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Linux / macOS
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Variables de Entorno / Environment Variables

```bash
cp .env.example .env
# Editar .env con tu configuración / Edit .env with your configuration
```

Para desarrollo local sin Docker, el sistema opera en modo eager con job store
en memoria. Redis es opcional.

For local development without Docker, the system runs in eager mode with an
in-memory job store. Redis is optional.

### Frontend (Next.js)

```bash
cd web
npm install
npm run dev  # http://localhost:3000
```

### Docker (stack completo / full stack)

```powershell
# Windows
.\run.ps1

# Linux / macOS
./run.sh
```

---

## Estructura del Proyecto / Project Structure

```
academic-automl-platform/
├── app/                  # Monolito composable (dev/tests)
├── packages/shared/      # Paquete compartido (automl_shared)
├── services/             # Microservicios
│   ├── gateway/          # Proxy HTTP, auth, rate limit
│   ├── ingestion/        # Upload y profiling de datasets
│   ├── ml-core/          # Entrenamiento y orquestación Celery
│   ├── explainability/   # Figuras SHAP
│   ├── exporter/         # Bundle científico (LaTeX, PDF, ZIP)
│   └── inference/        # Predicción
├── web/                  # Frontend Next.js 16
├── tests/                # Suite de tests
│   ├── unit/             # Tests unitarios
│   ├── integration/      # Tests de integración
│   └── compose/          # Tests Tier-2 (Docker stack)
├── data/                 # Datos runtime (gitignored)
├── docker-compose.yml    # Stack completo (11 servicios)
├── requirements.txt      # Dependencias producción
└── requirements-dev.txt  # Dependencias desarrollo
```

---

## Desarrollo / Development

### Modo de desarrollo local / Local development mode

El monolito componible (`app/`) re-exporta los paquetes de servicio y permite
desarrollo rápido sin Docker:

The composable monolith (`app/`) re-exports the service packages and enables
fast development without Docker:

```bash
uvicorn app.main:app --reload
```

### Microservicios / Microservices

Cada microservicio en `services/` tiene su propio `pyproject.toml` y `Dockerfile`.
Para trabajar en un servicio específico:

Each microservice in `services/` has its own `pyproject.toml` and `Dockerfile`.
To work on a specific service:

```bash
cd services/ml-core
pip install -e .
```

### Frontend

El frontend se comunica con el backend via rewrites de Next.js
(`API_ORIGIN` en `web/.env`). No se necesita CORS.

The frontend communicates with the backend via Next.js rewrites
(`API_ORIGIN` in `web/.env`). No CORS needed.

---

## Estilo de Código / Code Style

### Python

- **Linter:** Ruff (reglas: E, F, W, I, UP, B)
- **Line length:** 110 caracteres
- **Target:** Python 3.13

```bash
ruff check .           # Verificar / Check
ruff check --fix .     # Auto-fix
ruff format .          # Formatear / Format
```

### TypeScript / React (web/)

- **Linter:** ESLint (configurado en `web/eslint.config.mjs`)
- **Framework:** Next.js 16 App Router
- **UI:** shadcn/ui con Tailwind CSS

```bash
cd web
npm run lint
```

### Convenciones / Conventions

- No agregues comentarios a menos que se soliciten / Don't add comments unless
  requested
- Sigue los patrones existentes del codebase / Follow existing codebase patterns
- Usa type hints en Python / Use type hints in Python
- Prefiere composición sobre herencia / Prefer composition over inheritance

---

## Tests

### Tier 1: Suite completa (monolito) / Full suite (monolith)

```bash
# Windows
.venv\Scripts\python -m pytest tests -q -p no:warnings

# Linux / macOS
.venv/bin/python -m pytest tests -q -p no:warnings
```

### Tier 2: Smoke tests contra Docker stack / Smoke tests against Docker stack

```bash
docker compose --profile tier2 run --rm tier2
```

### Regenerar fixtures / Regenerate fixtures

```bash
python tests/fixtures/make_fixtures.py
```

**Antes de enviar un PR, asegúrate de que todos los tests pasan.**
**Before submitting a PR, make sure all tests pass.**

---

## Commits

Seguimos el formato [Conventional Commits](https://www.conventionalcommits.org/):

We follow the [Conventional Commits](https://www.conventionalcommits.org/)
format:

```
<tipo>(<scope>): <descripción corta>

feat(mlcore): agregar soporte para XGBoost
fix(exporter): corregir resolución de métrica primaria
refactor(ingestion): extraer profiling a módulo separado
docs: actualizar README con instrucciones de inference
test: agregar tests para inference service
chore: actualizar dependencias
```

**Tipos / Types:**

| Tipo / Type     | Descripción / Description                        |
|-----------------|--------------------------------------------------|
| `feat`          | Nueva funcionalidad / New feature                 |
| `fix`           | Corrección de bug / Bug fix                       |
| `refactor`      | Reestructuración sin cambio de comportamiento / Refactor without behavior change |
| `docs`          | Cambios en documentación / Documentation changes  |
| `test`          | Agregar o corregir tests / Add or fix tests       |
| `chore`         | Tareas de mantenimiento / Maintenance tasks       |
| `perf`          | Mejora de rendimiento / Performance improvement   |

---

## Pull Requests

1. Crea una rama desde `microservices` (o `main`):
   Create a branch from `microservices` (or `main`):

   ```bash
   git checkout -b feat/tu-feature microservices
   ```

2. Haz tus cambios con commits claros / Make your changes with clear commits
3. Corre los tests / Run the tests
4. Verifica el linter / Check the linter
5. Abre un PR con una descripción clara / Open a PR with a clear description

### Checklist del PR / PR Checklist

- [ ] Tests pasan localmente / Tests pass locally
- [ ] Ruff clean (Python) / ESLint clean (TypeScript)
- [ ] No se incluyen secrets o credenciales / No secrets or credentials included
- [ ] Documentación actualizada si aplica / Documentation updated if applicable
- [ ] Descripción clara del cambio / Clear description of the change
- [ ] Tipo de cambio marcado / Change type marked

---

## Reportar Bugs / Report Bugs

Usa el template de [Bug Report](https://github.com/Llamas126/academic-automl-platform/issues/new?template=bug_report.yml)
en GitHub Issues.

Use the [Bug Report](https://github.com/Llamas126/academic-automl-platform/issues/new?template=bug_report.yml)
template on GitHub Issues.

Incluye: descripción, pasos para reproducir, comportamiento esperado vs actual,
entorno (OS, Python, Docker), y logs si es posible.

Include: description, steps to reproduce, expected vs actual behavior, environment
(OS, Python, Docker), and logs if possible.

---

## Solicitar Funcionalidades / Request Features

Usa el template de [Feature Request](https://github.com/Llamas126/academic-automl-platform/issues/new?template=feature_request.yml)
en GitHub Issues.

Use the [Feature Request](https://github.com/Llamas126/academic-automl-platform/issues/new?template=feature_request.yml)
template on GitHub Issues.

Describe: qué problema resuelve, tu propuesta de solución, alternativas
consideradas, y el alcance (backend/frontend/infra).

Describe: what problem it solves, your proposed solution, alternatives considered,
and the scope (backend/frontend/infra).

---

## Preguntas / Questions

Usa el template de [Question](https://github.com/Llamas126/academic-automl-platform/issues/new?template=question.yml)
para preguntas sobre uso, configuración o arquitectura.

Use the [Question](https://github.com/Llamas126/academic-automl-platform/issues/new?template=question.yml)
template for questions about usage, configuration, or architecture.

---

## Seguridad / Security

**No abras issues públicas para vulnerabilidades de seguridad.**

**Do NOT open public issues for security vulnerabilities.**

Envía un email a **juanc-llamasc@qbitsglobal.com**. Ver
[SECURITY.md](SECURITY.md) para más detalles.

Send an email to **juanc-llamasc@qbitsglobal.com**. See
[SECURITY.md](SECURITY.md) for more details.

---

## Licencia / License

Al contribuir, aceptas que tus contribuciones sean licenciadas bajo la
[MIT License](LICENSE).

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE).
