# AIKI-HLA inference-only Docker image (CPU).
#
# Lightweight image for users who already have ESM-2 embeddings cached, or who
# only want to score (peptide, allele) pairs against the released 3-seed
# deployment ensemble. The 3 checkpoints (~142 MB total) are downloaded from
# the Zenodo concept DOI on first run and cached under /models.
#
# Published as: ghcr.io/aikium-public/aiki-hla:<version>
#               ghcr.io/aikium-public/aiki-hla:latest
#
# Usage:
#   docker run --rm -v aiki-hla-models:/models \
#       ghcr.io/aikium-public/aiki-hla:latest \
#       score --peptides peptides.csv --alleles alleles.csv --out predictions.csv
#
# For the heavier image that includes ESM-2 650M for on-the-fly embedding of
# novel peptides, see Dockerfile.full → ghcr.io/aikium-public/aiki-hla:<version>-full
FROM python:3.11-slim

# OCI labels for GHCR discovery
LABEL org.opencontainers.image.title="AIKI-HLA"
LABEL org.opencontainers.image.description="Open peptide-MHC predictor leading on novel-allele generalization, with the broadest open HLA coverage (inference-only, CPU)"
LABEL org.opencontainers.image.source="https://github.com/aikium-public/aiki-hla"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.authors="Venkatesh Mysore <venkatesh@aikium.com>"

WORKDIR /app

# Install CPU-only torch (smaller wheels) before the rest of the stack
RUN pip install --no-cache-dir \
    torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu

# Install aiki-hla + minimum runtime dependencies
COPY pyproject.toml README.md LICENSE ./
COPY aiki_hla/ aiki_hla/
RUN pip install --no-cache-dir .

# Persistent cache for downloaded checkpoints (mount with -v aiki-hla-models:/models)
ENV AIKI_MHC_CACHE=/models
VOLUME ["/models"]

# Non-root user for safer execution
RUN useradd -m -u 1000 aiki && chown -R aiki:aiki /app /models 2>/dev/null || true
USER aiki

ENTRYPOINT ["aiki-hla"]
CMD ["--help"]
