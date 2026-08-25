# Política de Seguridad / Security Policy

## Versiones Soportadas / Supported Versions

| Versión / Version | Soportado / Supported |
|-------------------|----------------------|
| 0.1.x             | :white_check_mark:   |

---

## Reportar Vulnerabilidades / Reporting Vulnerabilities

**No abras issues públicas para vulnerabilidades de seguridad.**

**Do NOT open public issues for security vulnerabilities.**

Si descubres una vulnerabilidad de seguridad en este proyecto, por favor envía un
correo electrónico directamente a **juanc-llamasc@qbitsglobal.com** con:

If you discover a security vulnerability in this project, please send an email
directly to **juanc-llamasc@qbitsglobal.com** with:

- **Descripción / Description:** Una descripción clara de la vulnerabilidad
- **Pasos para reproducir / Steps to reproduce:** Instrucciones detalladas
- **Impacto potencial / Potential impact:** Qué tan severo estimas que es
- **Versión afectada / Affected version:** Qué versión o commit
- **Sugerencia de fix (opcional) / Fix suggestion (optional):** Si tienes una
  propuesta de corrección

**Expected response timeline:**

- **Acknowledgment / Acuse de recibo:** Dentro de 48 horas / Within 48 hours
- **Initial assessment / Evaluación inicial:** Dentro de 5 días hábiles / Within
  5 business days
- **Resolution / Resolución:** Depende de la severidad / Depends on severity

Te mantendremos informado durante todo el proceso de resolución.

We will keep you informed throughout the resolution process.

---

## Medidas de Seguridad Implementadas / Security Measures

Este proyecto implementa múltiples capas de seguridad:

This project implements multiple security layers:

### Autenticación y Autorización / Authentication and Authorization

- **API Key authentication** (opt-in via `AUTH_ENABLED=true`)
- Constant-time comparison para prevenir timing attacks
- Keys configuradas vía variable de entorno `API_KEYS`

### Rate Limiting

- Sliding-window rate limiter (opt-in via `RATE_LIMIT_ENABLED=true`)
- Por API key o IP con fallback en memoria
- Responde HTTP 429 con header `Retry-After`

### Validación de Entrada / Input Validation

- IDs validados con regex `^[A-Za-z0-9_-]{1,64}$`
- Nombres de archivo saneados (path traversal prevention)
- Límites estructurales: `MAX_DATASET_ROWS`, `MAX_DATASET_COLUMNS`,
  `PREDICT_MAX_RECORDS`, `PREDICT_MAX_BODY_MB`
- Lectura de uploads por chunks con aborto temprano (413)

### Integridad de Artefactos / Artifact Integrity

- Modelos firmados con HMAC-SHA256 (`ARTIFACT_SIGNING_KEY`)
- Verificación de integridad al deserializar `.joblib`
- Modos: `off`, `if_present` (default), `strict`

### Cabeceras HTTP / HTTP Headers

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Referrer-Policy` restrictivo
- `Content-Security-Policy: default-src 'none'`
- HSTS opcional (`HSTS_ENABLED`)
- CORS restringido a orígenes en `CORS_ORIGINS`

### Contenedores / Containers

- Non-root user
- `no-new-privileges`
- `cap_drop ALL`
- Límites de memoria/CPU
- Redis con `--requirepass`
- Red dedicada (`automl` bridge)
- `beat` sin volumen de datos

---

## Buenas Prácticas para Contribuidores / Best Practices for Contributors

- Nunca commitees secrets, API keys, o credenciales
- No loguees datos sensibles en outputs de tests
- Usa variables de entorno para configuración sensible
- Reporta vulnerabilidades por email, no en issues públicas

- Never commit secrets, API keys, or credentials
- Don't log sensitive data in test outputs
- Use environment variables for sensitive configuration
- Report vulnerabilities via email, not in public issues
