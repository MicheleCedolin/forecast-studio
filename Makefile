PY ?= python3
export PYTHONPATH := src

.PHONY: test app data-check index hash template

test:            ## birim testleri (R ve torch gerektirmez)
	$(PY) -m pytest

app:             ## yerel arayüzü başlat (tarayıcıda açılır)
	$(PY) -m tsfstudio app

data-check:      ## data/tsf içeriğini katalogla karşılaştır
	$(PY) -m tsfstudio data-check

index:           ## veri özetlerini (_index.json) üret
	$(PY) -m tsfstudio index

hash:            ## veri parmak izi (SHA256SUMS.txt)
	$(PY) -m tsfstudio hash

template:        ## boş Excel şablonunu yaz
	$(PY) -m tsfstudio template forecast_studio_template.xlsx
