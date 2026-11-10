# Image de base Python
FROM python:3.13-slim

# Dossier de travail
WORKDIR /app

# Installer les dépendances
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copier l'API et le modèle exporté
COPY app.py .
COPY model/ model/

# Charger le modèle local
ENV MODEL_URI=model

# Port de l'application
EXPOSE 8000

# Démarrer FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
