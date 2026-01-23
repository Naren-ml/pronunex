# Pronunex API Endpoints Summary

**Total Endpoints: 20**

Base URL: `http://localhost:8000/api/v1/`

---

## Authentication (8 endpoints)

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/auth/register/` | User registration | No |
| POST | `/auth/login/` | JWT login, returns access & refresh tokens | No |
| POST | `/auth/logout/` | Blacklist refresh token | Yes |
| POST | `/auth/token/refresh/` | Refresh access token | No |
| GET | `/auth/profile/` | Get user profile | Yes |
| PUT | `/auth/profile/` | Update user profile | Yes |
| POST | `/auth/password/reset/` | Request password reset email | No |
| POST | `/auth/password/reset/confirm/` | Confirm password reset with token | No |
| POST | `/auth/password/change/` | Change password (logged in) | Yes |

---

## Library - Phonemes & Sentences (5 endpoints)

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/phonemes/` | List all 44 English phonemes | Yes |
| GET | `/phonemes/{id}/` | Get phoneme details (ARPAbet, IPA, tips) | Yes |
| GET | `/sentences/` | List practice sentences | Yes |
| GET | `/sentences/{id}/` | Get sentence with phoneme data | Yes |
| GET | `/sentences/recommend/` | Personalized sentence recommendations | Yes |

---

## Practice - Sessions & Assessment (6 endpoints)

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/sessions/` | List user's practice sessions | Yes |
| POST | `/sessions/` | Create new practice session | Yes |
| GET | `/sessions/{id}/` | Get session with attempts | Yes |
| POST | `/sessions/{id}/end/` | End session, calculate metrics | Yes |
| GET | `/attempts/` | List user's pronunciation attempts | Yes |
| GET | `/attempts/{id}/` | Get attempt details with errors | Yes |
| **POST** | `/assess/` | **Core pronunciation assessment** | Yes |

---

## Analytics - Progress Tracking (3 endpoints)

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/analytics/progress/` | Progress dashboard (scores, streaks, trends) | Yes |
| GET | `/analytics/phonemes/` | Per-phoneme analytics | Yes |
| GET | `/analytics/history/` | Daily progress history | Yes |

---

## Authentication Header

For authenticated endpoints, include:
```
Authorization: Bearer <ACCESS_TOKEN>
```

---

## Quick Reference

| Category | Count | Base Path |
|----------|-------|-----------|
| Authentication | 8 | `/api/v1/auth/` |
| Library | 5 | `/api/v1/phonemes/`, `/api/v1/sentences/` |
| Practice | 6 | `/api/v1/sessions/`, `/api/v1/attempts/`, `/api/v1/assess/` |
| Analytics | 3 | `/api/v1/analytics/` |
| **Total** | **20** | |
