# AI Coding Agent Instructions for Bread Checkout Software

## Overview
This repository contains the Bread Checkout Software, a project designed to model and predict bread-related data using machine learning workflows. The codebase is structured to support modular development, with clear separation of concerns across data processing, model training, and API services.

## Architecture
The project is organized into the following major components:

1. **Data Handling**:
   - Located in `src/datasets/`.
   - Key file: `bread_data.py` for loading and preprocessing bread-related datasets.

2. **Domain Models**:
   - Located in `src/domain/`.
   - Defines core entities such as `log_entity.py`, `map_bread.py`, and `prediction.py`.

3. **Modeling**:
   - Located in `src/model/`.
   - Key file: `bread_model.py` for defining and training machine learning models.

4. **Services**:
   - Located in `src/services/`.
   - Includes `meta_data_service.py` and `training_pipeline.py` for orchestrating workflows.

5. **API**:

## Developer Workflows

### Setting Up the Environment
1. Install `pipx` globally:
   ```sh
   pip install --user pipx
   ```
2. Install Poetry globally:
   ```sh
   pipx install poetry
   ```
3. Create a virtual environment:
   ```sh
   python -m venv ./venv
   ```
4. Activate the virtual environment and install dependencies:
   ```sh
   poetry install
   ```

### Running the Application
- The main entry point is `app.py`.
- Use `docker-compose-dev.yaml` for setting up development containers.

## Project-Specific Conventions
- Follow the modular structure for adding new components.
- Place reusable utilities in `helper_functions.py`.
- Store trained models in `ready_models/`.
- Use `src/const.py` for constants shared across modules.

## External Dependencies
- **Poetry**: Dependency management.
- **Docker**: Development environment setup.
- **MLflow**: Experiment tracking (artifacts stored in `mlruns/`).

---
This document is a work in progress. Update it as the project evolves.