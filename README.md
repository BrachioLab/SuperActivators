## Python Environment Setup
Run the following the setup and activate a Python development environment to run
the code:
```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision torchaudio
python -m pip install -r requirements.txt
python -m pip install -e .
```