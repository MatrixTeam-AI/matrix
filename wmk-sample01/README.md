# Setup

Setup and activate the virtual environment. Install dependencies.

```sh
python3.11 -m venv .venv 
source .venv/bin/activate
pip install -r requirements.txt
```

# Run

```sh
python main.py
```

Press G to make the screen green and verify that inputs work.

# Deploy

```sh
# Currently, only pip and Linux are supported. Windows and uv are on the roadmap.
wmk package --target . --name Build.zip --platform manylinux2014_x86_64
```

Upload the resulting Build.zip to the Build.zip file section. Under the Advanced
tab:

- Set Operating System as Linux
- Toggle on Enable OnDemand             # not visible
- Set Runtime as Python
- Set Run Command as `python main.py`   # not visible

It may be that these settings are not visible to you, contact support to edit
them if needed.
