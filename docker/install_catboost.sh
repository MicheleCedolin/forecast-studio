#!/usr/bin/env bash
# CatBoost R paketi CRAN'da değildir; resmi GitHub sürüm varlığından kurulur.
# Varlık adı sürümler arasında değişebildiğinden GitHub API'den KEŞFEDİLİR (adı elle kodlanmaz).
# Kurulamazsa (ör. ağ erişimi yok ya da sürümde ilgili mimari varlığı yok) imaj yine de oluşur; yalnızca R'daki CatBoost kullanılamaz ve
# `python -m tsfstudio doctor` bunu açıkça bildirir. Kurulumu zorunlu kılmak için: --build-arg CATBOOST_REQUIRED=1
set -u
VER="${CATBOOST_VERSION:?CATBOOST_VERSION gerekli}"
REQ="${CATBOOST_REQUIRED:-0}"
ARCH="$(uname -m)"

fail() {
  echo "UYARI: CatBoost R paketi kurulamadı: $1" >&2
  if [ "$REQ" = "1" ]; then exit 1; fi
  exit 0
}

# v1.2.10 varlıkları (doğrulandı): catboost-R-Linux-<v>.tgz, catboost-R-linux-x86_64-<v>.tgz, catboost-R-linux-aarch64-<v>.tgz
case "$ARCH" in
  x86_64|amd64)  PAT='catboost-R-linux-x86_64-'  ; FALLBACK='catboost-R-Linux-' ;;
  aarch64|arm64) PAT='catboost-R-linux-aarch64-' ; FALLBACK='' ;;
  *) fail "desteklenmeyen mimari ($ARCH)" ;;
esac

API="https://api.github.com/repos/catboost/catboost/releases/tags/v${VER}"
JSON="$(curl -fsSL --retry 3 --retry-delay 2 "$API")" || fail "GitHub API'ye ulaşılamadı ($API)"
URLS="$(printf '%s' "$JSON" | grep -o '"browser_download_url": *"[^"]*"' | sed 's/.*"\(http[^"]*\)"/\1/')"
URL="$(printf '%s\n' "$URLS" | grep -i "$PAT" | head -n 1)"
if [ -z "$URL" ] && [ -n "$FALLBACK" ]; then
  URL="$(printf '%s\n' "$URLS" | grep -i "$FALLBACK" | grep -vi 'aarch64\|arm' | head -n 1)"
fi
[ -n "$URL" ] || fail "v${VER} sürümünde $ARCH için R paketi varlığı bulunamadı"
echo "CatBoost R varlığı: $URL"

TMP="$(mktemp -d)"
curl -fsSL --retry 3 --retry-delay 2 -o "$TMP/catboost.tgz" "$URL" || fail "indirilemedi"
R CMD INSTALL --no-multiarch --no-test-load --no-staged-install "$TMP/catboost.tgz" || fail "R CMD INSTALL başarısız"
Rscript -e 'suppressPackageStartupMessages(library(catboost)); cat("catboost", as.character(packageVersion("catboost")), "yüklendi\n")' \
  || fail "paket yüklenemedi"
rm -rf "$TMP"
