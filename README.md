# Cisco Nexus 9000 Troubleshooting Assistant

A small, local Retrieval-Augmented Generation demo. It indexes an authorized
Cisco Nexus 9000 troubleshooting PDF in persistent ChromaDB, retrieves the
five most relevant passages, and uses Groq GPT-OSS 120B to answer in Gradio.
The local embedding model runs on CPU.

## Requirements

- Windows with Python 3.12 and internet access for package/model downloads and
  Groq inference.
- Your own Groq API key in a local `.env` file as `GROQ_API_KEY`.
- An authorized copy of the guide at
  `data/cisco_nexus_9000_troubleshooting_guide.pdf`.

The PDF and `.env` file are intentionally not included in this public
repository. Obtain the guide from an authorized Cisco source and place it at
the expected path. Never publish your API key.

This repository shares the application source; GitHub does not run the Gradio
app. Each person must run it locally with their own Groq API key and an
authorized copy of the guide.

## Setup and run (PowerShell)

From the project directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env and set your own GROQ_API_KEY value.
New-Item -ItemType Directory -Force data
# Place your authorized PDF at data\cisco_nexus_9000_troubleshooting_guide.pdf
python -c "import sys; print(sys.executable)"
python app.py
```

Open the local URL printed by Gradio (normally `http://127.0.0.1:7860`). Make
sure the printed Python executable is inside this project's `.venv`.
If `.venv` already exists, skip creating it. If Python 3.12 is not available
through `py -3.12`, use the Python 3.12 executable path from `py -0p`.

## PDF index

On the first launch, the app extracts page text, splits it into 512-character
chunks with 60-character overlap, embeds the chunks with
`sentence-transformers/all-MiniLM-L6-v2`, and writes a persistent index to
`chroma_db/`. Later launches reuse that index. If the guide is replaced, stop
the app, remove the local `chroma_db` directory, and restart to rebuild it.
Page references displayed in answers use the PDF's one-based page numbering.

## Tests

Run local checks with:

```powershell
python -m unittest discover -s tests -v
```

The app has been exercised with both a direct factual question and a broader
troubleshooting question. Both use the same top-five retrieval flow, and
unsupported information should be identified rather than guessed.
