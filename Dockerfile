# syntax=docker/dockerfile:1.7
# Forecast Studio — R (orijinal modeller) + Python tek imajda. Kullanıcının R kurmasına gerek yoktur.
#
# Yeniden üretilebilirlik: rocker/r-ver'in sabit etiketli (latest olmayan) sürümleri, CRAN'ı R sürümünün çıktığı
# tarihin Posit Package Manager anlık görüntüsüne bağlar; yani aynı R_VERSION => aynı R paket sürümleri.
ARG R_VERSION=4.5.1
FROM rocker/r-ver:${R_VERSION}

ARG CATBOOST_VERSION=1.2.10
ARG CATBOOST_REQUIRED=0
# Hangi Python ekstraları kurulsun? B aşamasında yalnızca test araçları; ileriki aşamalarda "dev,app,pyengine,deep".
ARG PY_EXTRAS=dev

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # BLAS/OpenMP iş parçacıkları: mclapply çatalları ile aşırı abonelik ve sayısal tekrarlanamazlığı önler
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    TSFSTUDIO_DATA_DIR=/app/data/tsf \
    TSFSTUDIO_RUNS_DIR=/app/runs

# --- sistem paketleri (Python + R paketlerinin çalışma zamanı kütüphaneleri)
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip curl ca-certificates \
      libcurl4-openssl-dev libssl-dev libxml2-dev libfontconfig1-dev libfreetype6-dev libpng-dev \
      libjpeg-dev libtiff-dev libharfbuzz-dev libfribidi-dev \
    && rm -rf /var/lib/apt/lists/*

# --- R paketleri (CRAN anlık görüntüsü; ikili paketler)
RUN install2.r --error --skipinstalled -n "$(nproc)" forecast jsonlite \
    && Rscript -e 'for (p in c("forecast","jsonlite")) cat(p, as.character(packageVersion(p)), "\n"); cat(R.version.string, "\n")'

# --- CatBoost R (GitHub sürüm varlığı; kurulamazsa uyarı verir, imaj yine oluşur)
COPY docker/install_catboost.sh /tmp/install_catboost.sh
RUN CATBOOST_VERSION=${CATBOOST_VERSION} CATBOOST_REQUIRED=${CATBOOST_REQUIRED} bash /tmp/install_catboost.sh && rm /tmp/install_catboost.sh

# --- Python ortamı
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[${PY_EXTRAS}]"

# --- Veri (büyük katman; kod katmanından ÖNCE değil SONRA kopyalanır ki kod değişince yeniden oluşmasın)
# Veriyi imaja gömmek istemezseniz docker-compose.yml içindeki bind mount satırını açın ve bu COPY'yi kaldırın.
COPY data/tsf /app/data/tsf

COPY tests ./tests
COPY docs ./docs
COPY scripts ./scripts

VOLUME ["/app/runs", "/app/uploads"]
EXPOSE 8501
# Arayüz yerel çalışır (python -m tsfstudio app; docs/APP.md). Docker imajı komut satırı içindir:
ENTRYPOINT ["python", "-m", "tsfstudio"]
CMD ["--help"]
