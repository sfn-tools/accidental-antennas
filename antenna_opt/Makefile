PYTHON ?= python3
ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

.PHONY: sim smoke campaign_dir5 campaign_dir24 campaign_omni campaign_all export

smoke:
	$(PYTHON) -m tests.smoke_test

sim:
	$(PYTHON) -m sim.run_one --model dir24 --params designs/params.json --quality fast

campaign_dir5:
	$(PYTHON) -m opt.run_campaign dir5 --refine-quality fast --refine-top 0 --no-pygmo

campaign_dir24:
	$(PYTHON) -m opt.run_campaign dir24 --refine-quality fast --refine-top 0 --no-pygmo

campaign_omni:
	$(PYTHON) -m opt.run_campaign omni

campaign_all:
	$(PYTHON) -m opt.run_campaign all

export:
	$(PYTHON) -m tools.export_geometry
