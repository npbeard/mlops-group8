FROM mambaorg/micromamba:1.5.10-jammy

WORKDIR /app

COPY --chown=$MAMBA_USER:$MAMBA_USER conda-lock.yml /tmp/conda-lock.yml
RUN micromamba create -y -n serving -f /tmp/conda-lock.yml && \
    micromamba clean --all --yes

COPY --chown=$MAMBA_USER:$MAMBA_USER src /app/src
COPY --chown=$MAMBA_USER:$MAMBA_USER config.yaml /app/config.yaml
COPY --chown=$MAMBA_USER:$MAMBA_USER README.md /app/README.md

ENV PYTHONPATH=/app
ENV PORT=8000

EXPOSE 8000

CMD ["micromamba", "run", "-n", "serving", "/bin/sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
