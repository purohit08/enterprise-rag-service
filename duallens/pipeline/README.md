# DualLens Pipeline (`.py` version of the notebook)

The DualLens analysis as one module per stage (it replaces the original
`DualLens_Analytics.ipynb`), so the flow is readable and each piece is
importable/testable. Built on **LangChain**
(`ChatAnthropic` + `OpenAIEmbeddings` + `Chroma`), with all prompts in
[`prompts.md`](prompts.md).

> This is the **fixed-pipeline** version: quant → qual → judge → fuse, run in a
> set order. The **agentic** version, where Claude chooses tools in a loop, is
> [`../agent.py`](../agent.py). Same two lenses, two different architectures — a
> useful contrast to be able to explain.

## Notebook section → file

| Notebook section | File | What it does |
|---|---|---|
| 1. Setup & imports | [`config.py`](config.py) | Models, paths, chunking; LangChain client factories |
| 2a/2b. Quantitative lens | [`quant.py`](quant.py) | 3-yr momentum + financial-metrics table (yfinance) |
| 3a-3c. Qualitative lens | [`qual.py`](qual.py) | Load → chunk → embed → Chroma → RAG answer |
| 3d. Evaluation | [`judge.py`](judge.py) | LLM-as-judge (groundedness + relevance) |
| 4. Fusion | [`fusion.py`](fusion.py) | Per-company summaries + Buy/Hold/Sell ranking |
| (whole notebook) | [`run.py`](run.py) | Runs every stage top-to-bottom |
| (all prompts) | [`prompts.md`](prompts.md) | The instructions — edit here, not in code |

## Run

```bash
# keys in .env at the repo root: ANTHROPIC_API_KEY + OPENAI_API_KEY
uv pip install -r duallens/requirements.txt
python -m duallens.pipeline.run
```

First run builds the vector store at `duallens/pipeline/chroma_pipeline/`
(gitignored); later runs reuse it.

## Prompts

`prompts.md` holds every prompt as a `## name` + fenced block. `prompts.py` loads
them; `render("rag_answer", context=..., question=...)` fills `$placeholders`.
Change wording in the markdown and the pipeline picks it up — no code edits.
