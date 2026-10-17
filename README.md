# BankChurners MLOps Pipeline

## Projekt célja

A projekt célja egy teljes MLOps workflow megvalósítása a BankChurners ügyfélelvándorlási (Customer Churn) adatállományon.

A rendszer képes:

- Adatelőkészítésre
- Feature engineeringre
- Modelltanításra
- Modellkiértékelésre
- MLflow alapú kísérletkövetésre
- Modellregisztrációra
- REST API alapú predikcióra
- EvidentlyAI monitoringra
- Streamlit dashboard megjelenítésre
- Docker alapú futtatásra
- GitHub Actions CI/CD ellenőrzésekre

---

# Quick Start

```bash
git clone https://github.com/YEQJ5C/BankChurners.git
cd BankChurners

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

docker compose up -d
```

## Elérhető szolgáltatások

- **MLflow:** http://localhost:5000
- **FastAPI:** http://localhost:8000/docs
- **Streamlit Dashboard:** http://localhost:8501

---

# MLOps architektúra

```text
BankChurners Dataset
          │
          ▼
 Data Preparation
          │
          ▼
 Feature Engineering
          │
          ▼
 Model Training
          │
          ▼
 MLflow Tracking
          │
          ▼
 Model Registry
          │
          ▼
 FastAPI Prediction Service
          │
          ▼
 Docker Deployment
          │
          ▼
 Streamlit Dashboard
          │
          ▼
 EvidentlyAI Monitoring
```

---

# Fejlesztési környezet

## Infrastrukturális környezet

- VMware Workstation 17 Player
- Ubuntu 22.04.5 LTS
- Python 3.10.12
- Docker 29.9.0
- Docker Compose 5.6.0
- Git 2.34.1
- VS Code Remote SSH

---

# Projektstruktúra

```text
BankChurners/
├── .github/
│   └── workflows/            # GitHub Actions CI/CD workflow-ok
│
├── data/                     # DVC által kezelt adatok
├── docs/                     # Dokumentáció és monitoring riportok
│   └── screenshots/          # Rendszerfutási és bizonyíték képernyőképek
├── metrics/                  # Modellkiértékelési metrikák
├── models/                   # Betanított modellek és artefaktok
├── src/                      # Az alkalmazás forráskódja
├── tests/                    # Pytest alapú automatizált tesztek
│
├── app.py                    # Streamlit dashboard
│
├── Dockerfile                # Docker image definíció
├── docker-compose.yml        # Konténerek indítása
│
├── dvc.yaml                  # DVC pipeline definíció
├── dvc.lock                  # DVC pipeline állapotfájl
├── params.yaml               # Modell- és pipeline-paraméterek
│
├── mlflow.db                 # MLflow backend adatbázis (SQLite)
│
├── Makefile                  # Automatizált fejlesztői feladatok
├── requirements.txt          # Python függőségek
│
├── pyproject.toml            # Projektkonfiguráció
├── pyrightconfig.json        # Pyright konfiguráció
│
├── .gitignore                # Git kizárási szabályok
├── .dockerignore             # Docker kizárási szabályok
├── .dvcignore                # DVC kizárási szabályok
├── .flake8                   # Flake8 konfiguráció
├── .pre-commit-config.yaml   # Pre-commit hook konfiguráció
│
├── setup_project.py          # Projekt inicializálása
├── run_check.py              # Környezetellenőrző script
│
└── README.md                 # Projektdokumentáció
```

---

# Virtuális környezet létrehozása

```bash
python -m venv venv
source venv/bin/activate
```

## Függőségek telepítése

```bash
pip install -r requirements.txt
```

---

# Tesztelés

## Egységtesztek futtatása

```bash
pytest
```

## Coverage riport

```bash
pytest --cov=src
```

A projekt a CI/CD quality gate által előírt 95%-os minimumot meghaladó, 100%-os tesztlefedettséget ér el.

---
# Makefile

A projekt a gyakori fejlesztői és MLOps feladatok automatizálására Makefile-t használ.

## Elérhető parancsok

### Környezet inicializálása

```bash
make init
```

Telepíti a projekt függőségeit a `requirements.txt` alapján.

### Környezet ellenőrzése

```bash
make check
```

Lefuttatja a környezet- és telepítésellenőrzést.

### Kódminőség ellenőrzése

```bash
make lint
```

Flake8 alapú kódstílus-ellenőrzés.

```bash
make typecheck
```

Pyright alapú statikus típuselemzés.

### Adatkezelés

```bash
make add-data
```

A nyers adatok DVC követésbe vétele.

### Adatelőkészítés

```bash
make prepare
```

Adatelőkészítési pipeline futtatása.

### Feature Engineering

```bash
make features
```

Feature engineering lépések futtatása.

### Modelltanítás

```bash
make train
```

A modell tanítása és MLflow kísérletkövetés.

### Modellkiértékelés

```bash
make evaluate
```

A modell kiértékelése és a metrikák mentése.

### Teljes MLOps Pipeline

```bash
make pipeline
```

A teljes DVC-alapú MLOps pipeline futtatása:

```bash
dvc repro
```

A parancs automatikusan végrehajtja az összes szükséges pipeline lépést.

### Tesztelés

```bash
make test
```

Pytest futtatása coverage méréssel.

### API indítása

```bash
make run
```

FastAPI szolgáltatás indítása fejlesztői módban.

## Példa workflow

```bash
make init
make check
make pipeline
make test
make run
```

A fenti parancsok végrehajtják a projekt teljes MLOps folyamatát a környezet ellenőrzésétől a pipeline futtatásán és tesztelésén át az API indításáig.

---

# MLflow

MLflow Tracking Server indítása:

```bash
mlflow ui
```

Elérhetőség:

```text
http://localhost:5000
```

Az MLflow segítségével nyomon követhetők:

- Modellparaméterek
- Metrikák
- Futtatások
- Modellverziók

---

# FastAPI

API indítása:

```bash
uvicorn src.api:app --reload
```

Swagger dokumentáció:

```text
http://localhost:8000/docs
```

Health endpoint:

```text
http://localhost:8000/health
```

---

# Streamlit Dashboard

Dashboard indítása:

```bash
streamlit run app.py
```

Elérhetőség:

```text
http://localhost:8501
```

A dashboard megjeleníti:

- Modell státusz
- Monitoring Engine státusz
- MLflow státusz
- CI/CD státusz
- Teszt quality gate állapot
- EvidentlyAI monitoring eredmények
- Data Drift riport

---

# EvidentlyAI Monitoring

Monitoring futtatása:

```bash
python -m src.monitoring
```

A monitoring artefakt helye:

```text
docs/drift_report.html
```

A riport a Streamlit dashboardon keresztül is megtekinthető.

---

# Docker

## Docker image építése

```bash
docker build -t bankchurners:latest .
```

## Docker image-ek listázása

```bash
docker images
```

## Konténer indítása

```bash
docker run -p 8000:8000 bankchurners:latest
```

## API ellenőrzése

Swagger dokumentáció:

```text
http://localhost:8000/docs
```

Health endpoint:

```bash
curl http://localhost:8000/health
```

Várható válasz:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

# CI/CD

A projekt GitHub Actions alapú Continuous Integration folyamatot használ.

Automatikusan fut:

- flake8
- pytest
- coverage ellenőrzés
- pyright

Quality Gate:

```text
Coverage >= 95%
```

Aktuális eredmény:

```text
Coverage = 100%
```

---

# Használt technológiák

- Python 3.10
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Streamlit
- Docker
- MLflow
- DVC
- EvidentlyAI
- Pytest
- Pyright
- GitHub Actions

---

# Szerző

**Lukács János (YEQJ5C)**

MSc Adattudomány

BankChurners MLOps Pipeline Projekt
