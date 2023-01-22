FROM python:3.7.7


# Installer les dépendances requises
# Copier les fichiers de l'application
COPY . /app
RUN pip install -r /app/requirements.txt

# Exposer le port 5058
EXPOSE 5058

# Définir le point d'entrée de l'application
CMD ["python", "/app/testVggFace_perso.py"]