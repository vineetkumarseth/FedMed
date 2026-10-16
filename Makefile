PY ?= python3
CONFIG ?= configs/default.yaml

.PHONY: help install keys test test-all baseline federate chaos audit api dashboard-install dashboard dashboard-build sweep clean

help:
	@echo "make install          install the Python package (+ dev tools)"
	@echo "make test             unit tests (seconds)"
	@echo "make test-all         unit + end-to-end federation tests (~2.5 minutes)"
	@echo "make federate         centralized baseline, then 3-hospital federation, export, audit"
	@echo "make chaos            same, but hospital 3 crashes in round 4 and rejoins 10 s later"
	@echo "make api              dashboard API on :8000 (serves the latest run)"
	@echo "make dashboard        React dashboard on :5173"
	@echo "make sweep            privacy/utility sweep over the DP noise level"

install:
	$(PY) -m pip install -e ".[dev,brats]"

keys:
	fedmed gen-certs --config $(CONFIG)
	fedmed gen-keys --config $(CONFIG)

test:
	$(PY) -m pytest

test-all:
	$(PY) -m pytest -m "slow or not slow"

baseline:
	fedmed baseline --config $(CONFIG)

federate:
	fedmed federate --config $(CONFIG) --keep-api

chaos:
	fedmed federate --config $(CONFIG) --chaos 3:4 --rejoin-delay 10 --keep-api

audit:
	fedmed audit --config $(CONFIG)

api:
	fedmed api --config $(CONFIG)

dashboard-install:
	cd dashboard && npm install

dashboard:
	cd dashboard && npm run dev

dashboard-build:
	cd dashboard && npm run build

sweep:
	$(PY) scripts/privacy_sweep.py --config $(CONFIG)

clean:
	rm -rf runs certs keys .pytest_cache dashboard/dist
