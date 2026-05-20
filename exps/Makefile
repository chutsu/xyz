BUILD_TARGETS := build_kalibr
BUILD_TARGETS += build_msckf-vio
BUILD_TARGETS += build_okvis
BUILD_TARGETS += build_orbslam3
BUILD_TARGETS += build_vins-fusion
BUILD_TARGETS += build_vins-mono

define build_docker
	@docker build \
		-t $(1) \
		-f Dockerfile.$(1) \
		--build-arg UID=$(shell id -u) \
		--build-arg GID=$(shell id -g) .
endef

define run_docker
	@xhost +local:docker
	@docker run -e DISPLAY \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v /tmp:/tmp \
		-v $(PWD)/resource:/home/docker/resource \
		--network="host" \
		-it --rm benchmark/$(1) \
		/bin/bash
endef

help:
	@echo "\033[1;34m[make targets]:\033[0m"
	@egrep -h '\s##\s' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; \
		{printf "\033[1;36m%-30s\033[0m %s\n", $$1, $$2}'

install_docker: ## Install Docker
	@bash scripts/install_docker.bash

build_all: $(BUILD_TARGETS) ## Build all

build_kalibr:  ## Build Kalibr
	$(call build_docker,kalibr)

build_msckf-vio:  ## Build MSCKF-VIO
	$(call build_docker,msckf-vio)

build_okvis:  ## Build OKVIS
	$(call build_docker,okvis)

build_orbslam3: ## Build ORBSLAM3
	$(call build_docker,orbslam3)

build_vins-fusion: ## Build VINS-Fusion
	$(call build_docker,vins-fusion)

build_vins-mono: ## Build VINS-Mono
	$(call build_docker,vins-mono)

run_kalibr: build_kalibr  ## Run Kalibr
	$(call run_docker,kalibr)

run_msckf-vio: build_msckf-vio  ## Run MSKCF-VIO
	$(call run_docker,msckf-vio)

run_okvis: build_okvis ## RUN OKVIS
	$(call run_docker,okvis)

run_orbslam3: build_orbslam3  ## Run ORBSLAM3
	$(call run_docker,orbslam3)

run_vins-fusion: build_vins-fusion  ## Run VINS-Fusion
	$(call run_docker,vins-fusion)

run_vins-mono: build_vins-mono  ## Run VINS-Mono
	$(call run_docker,vins-mono)
