# Frontend — academic-automl-platform

Interfaz web construida con **Next.js 16** (App Router, Tailwind CSS, shadcn/ui)
para la plataforma AutoML académica.

Web interface built with **Next.js 16** (App Router, Tailwind CSS, shadcn/ui) for
the academic AutoML platform.

## Funcionalidades / Features

- Asistente de entrenamiento: subida de dataset → configuración → lanzamiento
- Seguimiento de jobs en tiempo real
- Dashboard de resultados (métricas por fold con IC95%, comparativa de modelos,
  galería SHAP, citación BibTeX)
- Consola de inferencia (registros manuales o archivo CSV/XLSX con export CSV)
- Internacionalización ES/EN con selector en la cabecera

## Desarrollo / Development

```bash
npm install
npm run dev  # http://localhost:3000
```

Las peticiones a `/api/*` se proxean al backend vía rewrites (`API_ORIGIN`, por
defecto `http://localhost:8000`); no se necesita CORS.

API requests to `/api/*` are proxied to the backend via rewrites (`API_ORIGIN`,
default `http://localhost:8000`); no CORS needed.

## Stack

- **Framework:** Next.js 16 (App Router)
- **UI:** shadcn/ui + Tailwind CSS v4
- **State:** React Query (@tanstack)
- **Forms:** React Hook Form + Zod
- **Charts:** Recharts
- **i18n:** Custom (ES/EN, 265 translation keys)

## Más Información / More Info

- [README principal / Main README](../README.md)
- [Guía de Contribución / Contributing Guide](../CONTRIBUTING.md)
- [Licencia / License](../LICENSE)
