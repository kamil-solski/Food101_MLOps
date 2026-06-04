# Disclaimer
This project was initially inspired by Daniel Bourke's PyTorch Deep Learning series (https://github.com/mrdbourke/pytorch-deep-learning.git). While the overall system, pipeline design, and MLOps integrations are original and significantly expanded, one early notebook in this repository was adapted from concepts and patterns explored in his tutorials. Since we are dealing with image recognition here, our system will be limited to neural networks, as they work best for such tasks. So there is no room for classic ML methods here, and this involves decisions in system design.  

# MLOps based on Food101 data
Every end-to-end AI system is designed for a specific dataset and its characteristics. The dataset may grow, but what we want to get out of it should remain the same. However, we can simplify the system if we decide to include only a specific family of models (the system was not designed to be compatible with other methods). By making the system more specialized, as in this case, we make it easier to program and thus easier to understand. In other repositories, we will explore much more flexible systems.
In this project we will deal with real-time inference (because we expect low latency predictions). Even though it is not continuous real-time inference (with no downtime like in monitoring systems), it is still considered real-time. Even if user provides with two or more photos, if model process requests one by one and retrun immediately predictions for each, it is still real-time inference.
It is impossible to design an entire system in your head. Many functions and decisions regarding implementation are made during the course of the project, so it is worth starting from the ground up. In this course we will learn how to create and design AI systems in three levels of complexity. Why this way? The more you delve into the complexity of AI systems, the more you realize the trade-offs involved. Like it is not necessarily a bad thing, but it will have consequences that will become apparent later on, so you need to be aware of them (for example log with onnx and then register or leave pytorch format for registry and convert during inference run). By starting with simpler solutions, you will be able to grasp what it is all about, ask questions, and adopt optimal implementation strategies.

* Level 1 (foundational pipeline) - THIS LEVEL:
  - The skeleton of AI system - how system works?
  - Unit tests and simple integration tests
  - Containerization logic
  - Experiments tracking

* Level 2 (production oriented pipeline)
  - Database usage with data versioning
  - Model versioning
  - Shadow / A/B testing
  - Full CI/CD pipeline
  - Model and data monitoring (infernce time, data/concept drift) - with retraining mechanism (notifications not automatic)
  - Features stores

* Level 3 (Enterprise-grade, scalable, reproducible AI systems):
  - Kubernetes deployment
  - Ray and Anyscale integration
  - Horizontal and vertical scaling (Kubernetes Spark)
  - Canary deployment and rollback strategies
BONUS: custom created real-time monitoring dashboard

The course will be divided according to the challenges faced by Machine Learning Engineers when creating automated end-to-end systems. What compromises do they have to face, etc.? For each issue, an example solution will be provided and, if necessary, additional solutions will be described (in the form of code snippets from this repository).

# How to do end-to-end projects?:
1) create prototypes for training models pipeline and inference in jupyter notebooks
2) implement code to python files in project structure (with necessary modifications)
3) test training loop in python files (small dataset and few epochs) to see if there are no errors
4) create inference logic in python files
5) connect both training and inference logic (with best model) and check if broader pipeline works (from training to deployment on python files)
6) contenarize with docker
7) test containers
8) write proper tests starting from sanity tests (they should be imported from root_project/tests folder to proper files)

# End-to-End ML pipeline/AI System/ with MLE and MLOps Principles (Food-101) - Level 1
⚠️ This project is a work in progress. Tutorials and features are under active development. Stay tuned for updates and join me on my journey through Machine Learning Engineering!

This projects provides full ML pipeline (with MLOps principles) for food computer vision food classification, implemented to website. For tutorial purpose we will use two methods of deployment FastAPI and CI/CD to handle copying artifacts (onnx model), build docker image and run tests.

# Project requirements and installation guide:
- linux (tested on Ubuntu 22.04)
- docker
https://docs.docker.com/engine/install/ubuntu

- Nvidia GPU/s
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
We will download repo key, configure it (without experimental), update apt, install nvidia-container-toolkit and configure
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update

sudo apt install nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo nvidia-ctk runtime configure --runtime=containerd

sudo systemctl restart docker
```
Now test it - it should display nvidia-smi and check response time for 10 iterations, flops per interaction etc. (benchmark)
```bash
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi

docker run --rm -it --gpus=all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
```

- conda or any other venv (remember to switch off poetry default environment creation) for example miniconda:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh
```
- poetry (for testing outside docker)
```bash
pipx install poetry
poetry config virtualenvs.create false
```
Check if ~/.config/pypoetry/config.toml was created and have this content:
[virtualenvs]
create = false


Project structure:
 
```
/Food101_MLOps/ 
├── Data/                        # Raw, interim, and processed datasets
│   ├── raw/                     # Original data
│   ├── processed/                # Preprocessed datasets
│   └── interim/                  # Temporary datasets (optional)
│
├── notebooks/                    # Jupyter notebooks for experimentation, data exploration and also preprocessing. It is also a good place for feature engineering/selection and saving to files, so it could be later automated in e.g. cross-validation in python (not notebook) file.
│
├── src/                          # ML source code
│   ├── __init__.py
│   ├── data/                     # Data loading, preprocessing, augmentation
│   ├── models/                   # Model architectures (multiple CNNs, etc.)
│   ├── training/                 # Training logic, trainers, schedulers
│   ├── evaluation/               # Evaluation metrics plots (like Loss, Accuracy), Metrics computation (Recall, F1 etc.), Testing plots (Confusion matrices, ROC), Generating reports (training and testing) 
│   ├── serving/                  # ONNX export helpers, inference sanity tests
│   ├── utils/                    # Helper functions, common code
│   ├── cli.py                    # CLI entrypoint (train, export, evaluate)
│   └── config.yaml               # Hyperparameters, architecture selection, paths
│
├── experiments/                   # MLFlow tracking runs
│   ├── mlruns/                    # MLFlow local run storage
│   │   ├── experiment_id         # For example for food-101_30%_tr70_va15_te15_20250802. I use time stamp because mlflow has that annoying logic that if you delete some experiment you won't be able to use exact name again without deleting whole experiments directory. Logic implemented by prevents that.
│   │   │   ├── run_id            # combination of used fold, model architecture and its hyperparameters
│   │   │   │   ├── artifacts/
│   │   │   │   │   └── loss_accuracy_plot.png  # two plots overlay loss from train validation and accuracy from train and validation         
│   │   │   │   ├── metrics/
│   │   │   │   │   ├── train_accuracy
│   │   │   │   │   ├── train_loss
...
│   │   │   └── models             # logged models with mlflow for all runs in single experiment (with signature)
...
│   │   └── models                 # model registry
│   └── tracking.db                # (optional) Local SQLite DB for MLFlow
│
├── outputs/                        # local folder for artifacts from current run only. Remember since it is a temporary file it will be cleaned during next cli.py run
│   ├── checkpoints/                # Exported ONNX or PyTorch models
│   ├── logs/                       # Training logs, console outputs
│   ├── metrics/                    # saved metrics in json format
│   ├── predictions/                # Inference outputs (optional)
│   └── figures/                    # Evaluation plots 
│
├── tests/                          # All tests
│   ├── unit/                       # Unit tests for data, models, training
│   ├── integration/                # Integration tests
│   │   ├── test_exported_model.py  # Validate ONNX inference independently
│   │   ├── test_end_to_end.py      # Run website Docker + send HTTP request
│   │   ├── sample_image.jpg        # Test image for inference
│   │   ├── copy_model.py           # Copy ONNX to website project
│   │   └── start_website.py        # Spin up/down website Docker Compose
│   ├── pipeline/                   # full pipeline test
│   ├── regression/                 # compatibility between old and new code
│   ├── performance/
│   ├── smoke/
│   └── monitoring_alarming/
│
├── scripts/                        # DevOps & automation scripts
│   ├── deploy_model.sh             # Copy ONNX model to Flask app folder
│   ├── run_local_mlflow.sh         # Start MLFlow UI locally
│   └── entrypoint.sh               # Entrypoint script for Docker modes
│
├── platform/                      # containers, orchestration, CI/CD infra, monitoring hooks
│   ├── infrastructure/              # Infra and pipeline configs
│   │   ├── github-actions/          # GitHub Actions YAML workflows
│   │   │   ├── ml_pipeline.yml      # ML training + model deployment
│   │   ├── gitlab-ci/               # (Optional) GitLab CI pipeline configs
│   │   ├── .env
│   │   └── docker-compose.yml      # Compose for local orchestration
│   └── docker/                     # Dockerfiles (each multi-stage: builder + slim runtime)
│       ├── Dockerfile.train        # Training container (torch; CPU or CUDA via build args)
│       ├── Dockerfile.serve        # FastAPI inference container (ONNX Runtime, CPU, no torch)
│       └── Dockerfile.mlflow       # MLflow tracking UI (lightweight)
│
├── logs/                             # Local dev logs (optional gitignored)
│
├── inference_api                     # entire logic for program that is using our trained model (FastAPI)
│   ├── main.py                # FastAPI app + endpoints. Run this file for local testing just like cli.py for training
│   ├── inference/
│   │   ├── loader.py          # resolve & load model (MLFlow alias or local path)
│   │   ├── predictor.py       # preprocess -> infer -> postprocess
│   ├── config.yaml            # serving config
│   └── utils/
│       ├── helpers.py         # some helpers functions
│       └── postprocessing.py  # top-k, label mapping etc.
│
├── .env                       # Secrets, MLflow URIs, etc.
├── pyproject.toml             # deps + uv dependency groups (serve / train / jupyter / dev)
├── MLproject                  # MLflow reproducibility config
└── README.md
```

conda env is needed only for notebooks. It uses the same pyproject.toml that docker containers (later implementation) will be using.

### MLOps and ML models life cycle - human in the loop
I see here three main blocks. First is data preparation that could be done in notebooks. User here can: extract/select features, create setup for cross-validation etc. Second is when data is ready, running training and experiments. User here can: choose hyperparameters, change architectures, schedule learning, check training metrics and monitoring artifacts etc. Third, when models are saved is deployment, where the best model is selected converted to onnx and implemented into inference code. The tests are preformed and the implementation of the model and its inference are monitored. This is the end of one life cycle of AI systems, users here monitor and check results of entire pipeline (like Data Scientist, Data Engineers and ML Engineers), but people that edit logic of system on which others work are ML Engineers and if designed points of human interaction with the system are insufficient, ML Engineers modify the project code.

So, to sum up:
* first phase — editing notebooks and processing data. ML Engineer can create Python scripts based on notebooks to automate this part.
* second phase — defining experiments by specifying architectures, editing the config.yaml file, monitoring metrics, and tracking charts.
* third phase — model deployment. Running automation and tracking metrics relevant to the production environment.
Of course, at each of these stages, MLE can make adjustments to the system in accordance with requirements and broadly understood feedback from stakeholders, but the overall structure and interactions with the system remain more or less the same. This production environment can operate continuously. The development process must be consciously guided by humans (even for cyclical retraining, etc.).

This project involves inference of model on website with FastAPI. For simplicity lets host our website and Food101-MLOps project on the same machine. That means we will use docker-compose to orchestrate both website and Food101-MLOps. In real projects, this is not always the case. Sometimes the website and ML pipeline are located on different servers. In such cases, instead of compose, we use Kubernetes, for example.
Docker-compose file will contain only Dockerfile for ML pipeline and Website, but it could also other container like postgres database.

### Docker envs:
Each service has its own **multi-stage** Dockerfile (a `builder` stage with the
compiler toolchain + `uv`, then a clean slim `runtime` stage that only receives the
resolved `/opt/venv`). Combined with per-service dependency groups in `pyproject.toml`,
this means:
- smaller images — build tools and `uv` never ship; the runtime base is `python:3.10-slim`
- each image installs **only its own deps** (`--group serve` vs `--group train`); the serve
  image carries no torch at all, the CPU train image carries no CUDA libraries
- dependency manifests are copied before the source, so editing code does not bust the
  expensive dependency layer (faster rebuilds)
- `.dockerignore` keeps `Data/`, `experiments/`, notebooks and `.venv/` out of the build context

#### How to run:
There are three steps and places controlled by user to interact with project. When creating end-to-end systems, containerization is the final step once all scripts have been checked locally.

Place your dataset in the project's `Data/` folder. The repo root is bind-mounted into
the containers at `/app`, so `Data/` is automatically available at `/app/Data` — no `.env`
setup is required. (If your dataset lives elsewhere on disk, symlink it as `Data/`.)

Build Dockerfiles (while being inside project root folder):
1) 
```bash
cd platform/infrastructure
```
2) build the images (CPU by default):
```bash
docker compose build train inference mlflow
```
For a CUDA training image instead (inference stays CPU/ONNX):
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build train
```
3) because we will integrate with exteranl project flask we need to manually create network
```bash
docker network create shared_network 
```
In Dockerfile we execute custom scipt which specify MODE of run:
CMD ["bash", "scripts/entrypoint.sh"]

!Obviously you run docker compose commands from place where docker-compose.yml is located!

WARRNING!!! docker and host got completly separate logging, so for example model trained on host won't be used when running docker inference.

1. Data preparation - Prepare datasets by running the notebook locally or implementing the data preparation logic in Python scripts. The Subset_Food-101_generator.ipynb notebook can be run locally to create the required dataset structure.

2. Running training - when data is ready, user can add architectures, modify config.yaml etc. to customize training and experiments stage.
```bash
docker compose run --rm train  # this command will run train once 
docker compose stop train  # if previous command was executed in background
```

3. Viewing MLflow UI - Access experiment tracking and model artifacts:
```bash
docker compose up -d mlflow  # MLflow UI available at http://localhost:5000
docker compose stop mlflow   # stop when done
```

**Note**: All containers use bind mounts to share the experiments directory, so MLflow artifacts and experiments are accessible both in Docker and on the host filesystem.

Training outside docker containers (while being inside project root folder)
```bash
PYTHONPATH=. python src/cli.py
mlflow ui --backend-store-uri experiments/mlruns  # it uses localhost:5000
```

3. Deployment - when experiments are done, the best model is chosen and converted to onnx. Files to edit: 
Two strategies will be presented:
- FastAPI: for simplicity it will be local using other project simple flask website on local machine
- CI/CD: exporting to flask website but on remote server

```bash
docker compose up -d inference
docker compose stop inference
```

Testing inference of model on host (remember to run training on host first, then promote a version by setting the `champion` alias in the MLflow UI — the loader serves ONLY the `champion` and errors out if none is set. Run these commands from the project root folder)
```bash
python -m uvicorn inference_api.main:app --host 0.0.0.0 --port 8000 --reload
curl -s http://localhost:8000/health | jq
curl -s -X POST http://localhost:8000/predict -F "file=@Data/cannoli.png" | jq  # cannoli.png is just example image inside Data folder (just download any food image jpg or png, fix path and test inference). You should see classes and probabilities for them
```

To inference model on website clone the following repo: https://github.com/kamil-solski/Test_website.git and start website. Enter localhost:5002/food101 in browser.

System ML pipeline also supports simultaneous model training and serving, as well as Shadow or A/B testing implementation.

Why do we deploy each component in a separate Docker container rather than in a single monolith? Because these end-to-end system components should be treated as microservices. They exchange information to ensure the efficient and correct operation of the entire system, but they can operate independently of each other.
Separate containers allow us to add in Dockerfiles: non-interactive CI/CD jobs (train container), health checks endpoints for orchestration tools (serve container), add local directories for live editing (dev container). Considering that each container will be used with different frequency and independetly this separation optimize not only sizes, but also efficiency (only what is needed is built and run). Logging and monitoring is also separated this way.

### Adding new architectures
1) src/models - place new files.py with models architectures
2) update config.yaml - architectures


### Data
For educational purposes, we will try to simplify the process as much as possible while retaining the most important concepts of MLOps and MLE. We could, of course, add some embedding model to extract features from images (raw image data is too large) and store them in a database, but this would have additional consequences that would increase the complexity of the project (the use of a database must be justified), e.g.:
- Increasing the complexity of the data flow to provide additional functionality — data/concept drift while maintaining optimal performance

- Introducing dependencies on a persistent storage layer (e.g., a relational or vector database), which requires provisioning, configuration, and integration in both development and production environments.

- Requiring learners to understand embedding models (e.g., CLIP or ResNet), their inference pipelines, and embedding dimensionality — potentially before they've fully grasped the basics of preprocessing and model training.

- Adding the need for vector similarity search, schema versioning, and efficient storage formats — which are advanced concerns that can be explored later.

- Making observability and debugging more difficult, as intermediate representations (like embeddings) are not as intuitively inspectable as raw labels or predictions.

Instead, this tutorial introduces these ideas gradually, and focuses first on getting learners comfortable with the core building blocks: reproducible preprocessing, model training, evaluation, and CI/CD. As learners progress, they’ll be better equipped to appreciate when and why such advanced features like embedding databases are necessary.

Although it could happen for example when user will provide image in negative and then model will have problem classifying. When user provide: 
* images in negative or darker than model is able to detect correctly - we got data drift
* images that are out of scope of known classes - we got concept drift
But that is topic for another lesson - "HOW WE COULD MONITOR DATA DURING INFERNCE?".

Here there is simple data versioning in a form of folders naming. Because of that we will omit data versioning, but for advanced pipelines with databases we should implement it. Because there is no data versioning model versioning doesn't make sense either. Because there are no models to choose from shadow or A/B testing is also pointless. All these simplifications are aimed at make the pipeline easier to understand, so that the operation diagram is not complicated with many branches and loops.

#### Cross validation and hyperparameters automation
We use K-Fold Cross-validation during data-prepraration to get dataset variations in folds folders. Dataloaders are running for those folders. This way we will treat each run as specifc combination of hyperarameters and model architecture. So, we can easily compute number of runs (each with own saved model) per single experiment: 

num_architectures * num_folds * num_combinations

But, at the end of training we should have two best models for each architecture, so we have to get averages per hyperparameter combination across folds (for each architecture) pick the best based on values, depending on what metric it is (loss minimization, AUC maximization, accuracy, precision, etc.). Here is an example:

|  fold0  |  fold1  |  fold2  |  average  |
| ------- | ------- | ------- | --------- |
|  comb1  |  comb1  |  comb1  |   avg1    |
|  comb2  |  comb2  |  comb2  |   avg2    |
|  comb3  |  comb3  |  comb3  |   avg3    |

I decided to manually implement hyperparamter automation (and not using GridSearch), because it will give us flexibility and control for future implementations. Understanding how algorithms like GridSearch work is essetial when working with end-to-end automated AI system.

#### Loop Order Design Decision
The training pipeline uses a specific nested loop order for optimal memory management and experiment organization:

```python
for arch_name in config["architectures"]:        # Outer loop
    for hu in config["hidden_units"]:            # Middle loop  
        for lr in config["learning_rates"]:      # Inner loop
            for fold in folds:                   # Innermost loop
                # Train model on this fold
```

**Why this order matters:**

1. **Memory Efficiency**: 
   - Dataloaders are created once per hyperparameter combination, not per fold
   - Better memory cleanup between hyperparameter combinations
   - Prevents memory accumulation across different configs

2. **Logical Experiment Grouping**:
   - All folds for the same hyperparameter combination are trained consecutively
   - Easier to compare results across folds for the same config
   - Better MLflow experiment organization and visualization

3. **Resource Management**:
   - Predictable memory usage patterns
   - Can implement early stopping based on cross-validation results
   - Better GPU memory management with proper cleanup cycles

4. **Debugging and Monitoring**:
   - Clear progress indication: "Training Arch1+Config1 across all folds"
   - Easier to identify which hyperparameter combination is causing issues
   - Better error isolation and recovery

**Alternative (problematic) order:**
```python
for fold in folds:                    # Would be outer loop
    for arch_name in config["architectures"]:
        for hu in config["hidden_units"]:
            for lr in config["learning_rates"]:  # Would be inner loop
```

This would cause:
- Memory accumulation (dataloaders kept in memory across all configs)
- Poor experiment organization in MLflow
- Difficult debugging and monitoring
- Inefficient resource usage

### Models
We desgin model architectures and place them inside src/model directory (as python files). From there, they will be used to take hyperparameter combination and run on specific fold.

--onnx_registry.py:
It does make sense to save only the best model to onnx and then register it. We find the best model, then log it as onnx and register it as a new version of "best_model".

At Level 1 we deliberately do NOT promote models automatically. Choosing which version serves production (the "champion") is a human decision, so we leave it to the user via the MLflow UI: open Models -> best_model -> pick a version -> add the alias `champion`. The serving API loads ONLY the `champion` version — there is no fallback to challenger or latest. If no `champion` alias exists, the inference service fails on startup with a clear error in the container logs, prompting you to promote a model first.

Automatic promotion, A/B and Shadow testing (challenger/prev_champion aliasing) are introduced in Level 2, where model versioning makes them meaningful.

### Trainer Architecture
The project uses a flexible, class-based trainer architecture that makes it easy to extend and maintain:

**Key Components:**
- `BaseTrainer`: Abstract base class defining the trainer interface
- `PyTorchTrainer`: Concrete implementation for PyTorch neural networks
- Legacy function-based API maintained for backward compatibility

**Benefits:**
- **Flexibility**: Easy to add new model architectures and training frameworks
- **Maintainability**: Centralized training logic in trainer classes
- **Testability**: Each trainer can be tested independently
- **Device Management**: Automatic handling of CUDA/CPU devices
- **Backward Compatibility**: Existing code continues to work without changes

**Usage:**
```python
# New OOP approach (recommended)
from src.training.engine import PyTorchTrainer

trainer = PyTorchTrainer(model, loss_fn, optimizer, device='cuda')
train_loss, train_acc = trainer.train_step(train_dataloader)
val_loss, val_acc = trainer.eval_step(val_dataloader)

# Legacy function-based approach (still supported)
from src.training import train_step, val_step

train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer)
val_loss, val_acc = val_step(model, val_dataloader, loss_fn)
```

For detailed usage examples and extending the architecture, see `docs/trainer_usage_guide.md` and run `examples/trainer_demo.py`:
```bash
cd Food101_MLOps-Lvl_1
conda activate Food101
PYTHONPATH=. python examples/trainer_demo.py
```

### Metrics and artifacts
For mlflow each run is combination of fold and hyperparameters with own saved model and plots (loss and accuracy overlay) at the end. I defined an experiment as an entire dataset because it is a constant part of the entire experiment. Its specific folds, model architectures and theirs hyperparameters may change, but the entire dataset used remains the same. 
All artifacts from current run are saved first to outputs folder and then to proper folder in mlruns (it is main storage for artifacts assigned for their runs). Outputs folder contains only artifacts from current experiment. 

On level 1 of project complexity we won't use database for mlflow, since we don't even use it for data.

### Inference
To check how it look like localy you can download Test Website repo that uses flask and nginx. For inference OpenCV is used because it got good support for ONNX models (they use optimized runtimes), because inference efficiency in many cases is crucial. 

### End
Once you understand how the code works and why it was designed this way, you can move on to analyzing my other projects, where the process is more practical and complete (it creates the life cycle of a working artificial intelligence system).

### Troubleshooting
* If you make chages, especially with folders names of mlflow or permmissions changeed -> rm -rf /outputs /experiments (while being in root). Also to fix compatibility issues clean volume for docker (docker compose down -v). This will obviously remove all saved runs (with artifacts and models).