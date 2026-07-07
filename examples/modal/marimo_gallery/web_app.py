from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import marimo
import markdown
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
DEFAULT_CONTENT_DIR = ROOT / "content"
DEFAULT_NOTEBOOKS_DIR = ROOT / "notebooks"
DEFAULT_EXAMPLE_ASSETS_DIR = ROOT.parents[1] / "04_episode_operations" / "hand_tracking"


@dataclass(frozen=True)
class GalleryItem:
    title: str
    label: str
    href: str
    description: str
    image: str | None = None


GALLERY_ITEMS = [
    GalleryItem(
        title="DROID episode index",
        label="DROID",
        href="/demos/droid-kitchen",
        description=(
            "Read public DROID metadata with Daft, filter episode rows, and "
            "inspect the query plan in an embedded Marimo app."
        ),
    ),
    GalleryItem(
        title="EgoDex hand tracking",
        label="Hand tracking",
        href="/demos/egodex-hands",
        description=(
            "Run the MediaPipe hand tracker locally, render keypoints on frames, "
            "and compare against EgoDex ground truth."
        ),
        image="/_example_assets/demo_keypoints.png",
    ),
    GalleryItem(
        title="Policy evals: failure mining",
        label="Policy evals",
        href="/demos/policy-evals",
        description=(
            "Real benchmark rollouts in-repo: OpenVLA 84% vs VLA-JEPA 99%, every "
            "failure labeled from step signals (16 of 17 are re-grasp loops)."
        ),
    ),
    GalleryItem(
        title="Pose features and scenario segments",
        label="Pose",
        href="https://github.com/Eventual-Inc/daft-physical-ai/tree/main/examples/03_transforms",
        description=(
            "Pure-NumPy pose tracks per episode on a public EgoDex sample - curl, "
            "pinch, palm orientation - with grasp/lift segments stitched in time."
        ),
    ),
    GalleryItem(
        title="Curate and hand off to training",
        label="Curation",
        href="https://github.com/Eventual-Inc/daft-physical-ai/tree/main/examples/06_writing_data",
        description=(
            "Motion-trimmed SFT views, preference pairs from the policy comparison, "
            "and to_torch_dataloader batches: (64, 7) actions, (64, 8) states."
        ),
    ),
    GalleryItem(
        title="Try it on your own data",
        label="Multibase",
        href="https://eventual.ai/multibase",
        description=(
            "Keep public examples small, then route proprietary robot datasets "
            "to Multibase for access control and hosted workflows."
        ),
    ),
]


NAV_ITEMS = [
    ("Getting Started", "/"),
    ("Demos", "/demos"),
    ("Examples", "/examples"),
]

TOP_NAV_ITEMS = [
    *NAV_ITEMS,
    ("Use Your Own Data", "https://eventual.ai/multibase"),
]

DOC_NAV_ITEMS = [
    *NAV_ITEMS,
    ("DROID episode index", "/demos/droid-kitchen"),
    ("EgoDex hand tracking", "/demos/egodex-hands"),
    ("Policy evals", "/demos/policy-evals"),
]

DEMO_TOPICS = [
    (
        "Running pipelines",
        "Move from local iteration to hosted execution without changing the notebook shape.",
        "/demos/droid-kitchen",
    ),
    (
        "Reading data",
        "Load robot datasets, metadata, videos, and table assets into Daft.",
        "/demos/droid-kitchen",
    ),
    (
        "Episode data",
        "Inspect episode rows, frame media, task fields, and success labels.",
        "/demos/policy-evals",
    ),
    (
        "Transforms",
        "Filter, join, type, and enrich robotics data with Daft expressions.",
        "/demos/droid-kitchen",
    ),
    (
        "Episode operations",
        "Annotate, trim, score, and track signals across episodes.",
        "/demos/egodex-hands",
    ),
    (
        "Inference",
        "Run models over images, video-derived rows, metadata, and structured columns.",
        "/demos/egodex-hands",
    ),
    (
        "Writing data",
        "Persist annotated datasets for training and downstream analysis.",
        "/demos/policy-evals",
    ),
    (
        "Policy evals",
        "Reproduce benchmark runs, compare policies on the same specs, and mine failures.",
        "/demos/policy-evals",
    ),
]


def render_markdown(path: Path) -> str:
    return markdown.markdown(
        path.read_text(),
        extensions=["fenced_code", "tables", "sane_lists"],
    )


def gallery_cards() -> str:
    cards = []
    for item in GALLERY_ITEMS:
        media = (
            f'<img src="{item.image}" alt="" loading="lazy" />'
            if item.image
            else '<div class="card-visual" aria-hidden="true"></div>'
        )
        external = item.href.startswith("https://")
        target = ' target="_blank" rel="noreferrer"' if external else ""
        cards.append(
            f"""
            <a class="gallery-card" href="{item.href}"{target}>
              {media}
              <span>{item.label}</span>
              <strong>{item.title}</strong>
              <p>{item.description}</p>
            </a>
            """
        )
    return "\n".join(cards)


def demo_topic_cards() -> str:
    return "\n".join(
        f"""
        <a class="topic-card" href="{href}">
          <span>{title}</span>
          <p>{description}</p>
        </a>
        """
        for title, description, href in DEMO_TOPICS
    )


def nav_html(active: str, *, topbar: bool = False) -> str:
    links = []
    items = TOP_NAV_ITEMS if topbar else DOC_NAV_ITEMS
    for label, href in items:
        aria = ' aria-current="page"' if href == active or (active.startswith(href) and href != "/") else ""
        external = href.startswith("https://")
        target = ' target="_blank" rel="noreferrer"' if external else ""
        class_name = ' class="nav-cta"' if topbar and external else ""
        links.append(f'<a href="{href}"{class_name}{aria}{target}>{label}</a>')
    return "\n".join(links)


def page_shell(
    *,
    title: str,
    active: str,
    body: str,
    eyebrow: str,
    summary: str,
    mode: Literal["docs", "wide"] = "docs",
    aside: str = "",
) -> HTMLResponse:
    aside_html = aside or """
      <div class="toc-card">
        <span>Workflow</span>
        <a href="/demos">Browse demos</a>
        <a href="/examples">Browse examples</a>
        <a href="/demos/droid-kitchen">Run DROID notebook</a>
        <a href="https://eventual.ai/multibase" target="_blank" rel="noreferrer">Use your own data</a>
      </div>
    """
    html = f"""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title} · daft-physical-ai</title>
    <style>{STYLE}</style>
  </head>
  <body>
    <header class="topbar">
      <a class="brand" href="/">daft-physical-ai</a>
      <nav aria-label="Main navigation">{nav_html(active, topbar=True)}</nav>
    </header>
    <main class="layout {mode}">
      <aside class="sidebar">
        <div class="sidebar-title">Docs</div>
        {nav_html(active)}
      </aside>
      <article class="content">
        <section class="page-hero">
          <p class="eyebrow">{eyebrow}</p>
          <h1>{title}</h1>
          <p>{summary}</p>
        </section>
        {body}
      </article>
      <aside class="right-rail">{aside_html}</aside>
    </main>
  </body>
</html>
"""
    return HTMLResponse(html)


def notebook_frame() -> str:
    return """
    <section class="notebook-panel" aria-label="Embedded Marimo notebook">
      <div class="notebook-head">
        <div>
          <span>Live notebook</span>
          <strong>DROID episode index</strong>
        </div>
        <a href="/_marimo/droid-kitchen/" target="_blank" rel="noreferrer">Open full screen</a>
      </div>
      <iframe
        src="/_marimo/droid-kitchen/"
        title="DROID episode index Marimo notebook"
        loading="lazy"
      ></iframe>
    </section>
    """


def egodex_panel() -> str:
    return """
    <section class="example-panel" aria-label="EgoDex hand tracking output">
      <img src="/_example_assets/demo_keypoints.png" alt="EgoDex ground truth and predicted hand keypoints" />
      <div>
        <span>Local demo</span>
        <strong>Rendered output is committed</strong>
        <p>
          The demo markdown, notebook, script, and keypoint image live in
          <code>examples/04_episode_operations/hand_tracking/</code>. Run it
          locally when you want fresh inference.
        </p>
        <div class="inline-actions">
          <a href="/_example_assets/demo.md">Read markdown</a>
          <a href="/_example_assets/demo.py">Open script</a>
          <a href="/_example_assets/demo.ipynb">Download notebook</a>
        </div>
      </div>
    </section>
    """


def create_app(
    *,
    content_dir: str | Path = DEFAULT_CONTENT_DIR,
    notebooks_dir: str | Path = DEFAULT_NOTEBOOKS_DIR,
    example_assets_dir: str | Path | None = DEFAULT_EXAMPLE_ASSETS_DIR,
) -> FastAPI:
    content_path = Path(content_dir)
    notebooks_path = Path(notebooks_dir)
    assets_path = Path(example_assets_dir) if example_assets_dir else None

    web = FastAPI(title="daft-physical-ai demos")

    if assets_path and assets_path.exists():
        web.mount("/_example_assets", StaticFiles(directory=assets_path), name="example_assets")

    marimo_server = marimo.create_asgi_app().with_app(
        path="/droid-kitchen",
        root=str(notebooks_path / "droid_kitchen.py"),
    )
    web.mount("/_marimo", marimo_server.build())

    @web.get("/", response_class=HTMLResponse)
    async def home() -> HTMLResponse:
        body = f"""
        <section class="feature-grid">
          <a href="/demos"><span>Demos</span><strong>Workflow topics</strong><p>Reading data, transforms, episode operations, inference, writing outputs, and policy evals.</p></a>
          <a href="/examples"><span>Examples</span><strong>The loop, measured</strong><p>Numbered 01-08 recipes running on real data committed in the repo.</p></a>
          <a href="https://eventual.ai/multibase" target="_blank" rel="noreferrer"><span>Private data</span><strong>Use your own data</strong><p>Route proprietary datasets to the managed Multibase workflow.</p></a>
        </section>
        <section class="doc-body">{render_markdown(content_path / "index.md")}</section>
        """
        return page_shell(
            title="Getting Started",
            active="/",
            eyebrow="Quick start",
            summary=(
                "Physical-AI data curation and policy evals on Daft - the loop from raw "
                "robot datasets to the training handoff, runnable offline on data in the repo."
            ),
            body=body,
            mode="wide",
        )

    @web.get("/demos", response_class=HTMLResponse)
    async def demos() -> HTMLResponse:
        body = f"""
        <section class="topic-grid">{demo_topic_cards()}</section>
        <section class="feature-grid">
          <a href="/demos/droid-kitchen"><span>Live demo</span><strong>DROID episode index</strong><p>Read DROID metadata, filter successful episodes, and inspect a Daft query plan.</p></a>
          <a href="/demos/egodex-hands"><span>Local demo</span><strong>EgoDex hand tracking</strong><p>Run MediaPipe locally and compare predictions against EgoDex ground truth.</p></a>
          <a href="/demos/policy-evals"><span>Analysis demo</span><strong>Policy evals</strong><p>Compare policies on the same benchmark specs and label re-grasp loops with a Daft scan.</p></a>
        </section>
        <section class="doc-body">{render_markdown(content_path / "demos.md")}</section>
        """
        return page_shell(
            title="Demos",
            active="/demos",
            eyebrow="Workflow topics",
            summary="Executable guides for robotics data work: reading data, transforms, episode operations, inference, and writing data.",
            body=body,
            mode="wide",
        )

    @web.get("/examples", response_class=HTMLResponse)
    async def examples() -> HTMLResponse:
        body = f"""
        <section class="gallery-grid">{gallery_cards()}</section>
        <section class="doc-body">{render_markdown(content_path / "examples.md")}</section>
        """
        return page_shell(
            title="Examples",
            active="/examples",
            eyebrow="Concrete workflows",
            summary="End-to-end physical-AI examples that show the input data, operation, and result.",
            body=body,
            mode="wide",
        )

    @web.get("/gallery")
    async def old_gallery() -> RedirectResponse:
        return RedirectResponse("/examples", status_code=307)

    @web.get("/droid")
    async def old_droid() -> RedirectResponse:
        return RedirectResponse("/demos/droid-kitchen", status_code=307)

    @web.get("/demos/droid-kitchen", response_class=HTMLResponse)
    async def droid() -> HTMLResponse:
        body = f"""
        <section class="doc-body">{render_markdown(content_path / "droid.md")}</section>
        {notebook_frame()}
        """
        return page_shell(
            title="DROID episode index",
            active="/demos",
            eyebrow="Embedded notebook",
            summary="A regular documentation page with a Marimo app embedded for the live Daft workflow.",
            body=body,
            mode="wide",
            aside="""
            <div class="toc-card">
              <span>On this page</span>
              <a href="/_marimo/droid-kitchen/" target="_blank" rel="noreferrer">Notebook full screen</a>
              <a href="https://eventual.ai/multibase" target="_blank" rel="noreferrer">Use your own data</a>
            </div>
            """,
        )

    @web.get("/demos/egodex-hands", response_class=HTMLResponse)
    async def egodex() -> HTMLResponse:
        body = f"""
        {egodex_panel()}
        <section class="doc-body">{render_markdown(content_path / "egodex.md")}</section>
        """
        return page_shell(
            title="EgoDex hand tracking",
            active="/demos",
            eyebrow="Local demo",
            summary="A CPU MediaPipe hand-tracking walkthrough that can run locally without Modal.",
            body=body,
            mode="wide",
            aside="""
            <div class="toc-card">
              <span>Artifacts</span>
              <a href="/_example_assets/demo.md">Rendered markdown</a>
              <a href="/_example_assets/demo.py">Script</a>
              <a href="/_example_assets/demo.ipynb">Notebook</a>
              <a href="https://eventual.ai/multibase" target="_blank" rel="noreferrer">Use your own data</a>
            </div>
            """,
        )

    @web.get("/demos/failure-modes")
    async def old_failure_modes() -> RedirectResponse:
        return RedirectResponse("/demos/policy-evals", status_code=307)

    @web.get("/demos/policy-evals", response_class=HTMLResponse)
    async def policy_evals() -> HTMLResponse:
        body = f"""
        <section class="doc-body">{render_markdown(content_path / "policy_evals.md")}</section>
        """
        return page_shell(
            title="Policy evals",
            active="/demos",
            eyebrow="Benchmark analysis",
            summary="A CPU-only demo that writes canonical rollout rows, compares policies with Daft, and labels re-grasp failures.",
            body=body,
            mode="wide",
            aside="""
            <div class="toc-card">
              <span>Artifacts</span>
              <a href="/examples">Examples</a>
              <a href="https://eventual.ai/multibase" target="_blank" rel="noreferrer">Use your own data</a>
            </div>
            """,
        )

    return web


STYLE = """
:root {
  color-scheme: light;
  --bg: #fbfbf8;
  --paper: #ffffff;
  --ink: #111411;
  --muted: #5f6761;
  --line: #dfe5dd;
  --line-strong: #c8d2c7;
  --accent: #4e7b43;
  --accent-2: #b24f3b;
  --code-bg: #f2f4ef;
  --shadow: 0 18px 48px rgb(31 41 32 / 0.08);
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
a { color: inherit; }
.topbar {
  position: sticky;
  top: 0;
  z-index: 20;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  min-height: 60px;
  padding: 0 28px;
  border-bottom: 1px solid var(--line);
  background: rgb(251 251 248 / 0.92);
  backdrop-filter: blur(14px);
}
.brand {
  font-weight: 760;
  text-decoration: none;
  letter-spacing: 0;
}
.topbar nav,
.sidebar {
  display: flex;
  gap: 6px;
}
.topbar nav a,
.sidebar a,
.toc-card a {
  border-radius: 6px;
  color: var(--muted);
  font-size: 14px;
  text-decoration: none;
}
.topbar nav a {
  padding: 7px 10px;
}
.topbar nav a[aria-current="page"],
.sidebar a[aria-current="page"],
.toc-card a:hover,
.sidebar a:hover,
.topbar nav a:hover {
  color: var(--ink);
  background: var(--code-bg);
}
.topbar nav a.nav-cta {
  color: #fff;
  background: var(--accent);
}
.topbar nav a.nav-cta:hover {
  color: #fff;
  background: #3f6936;
}
.layout {
  display: grid;
  grid-template-columns: 220px minmax(0, 760px) 220px;
  gap: 32px;
  max-width: 1320px;
  margin: 0 auto;
  padding: 36px 28px 80px;
}
.layout.wide {
  grid-template-columns: 220px minmax(0, 940px) 220px;
}
.sidebar,
.right-rail {
  position: sticky;
  top: 84px;
  align-self: start;
}
.sidebar {
  flex-direction: column;
}
.sidebar-title,
.toc-card span {
  margin-bottom: 8px;
  color: var(--accent);
  font: 760 12px/1.2 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.sidebar a,
.toc-card a {
  display: block;
  padding: 7px 9px;
}
.toc-card {
  padding: 14px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: rgb(255 255 255 / 0.72);
}
.content {
  min-width: 0;
}
.page-hero {
  padding: 20px 0 34px;
  border-bottom: 1px solid var(--line);
}
.eyebrow {
  margin: 0 0 12px;
  color: var(--accent);
  font: 760 12px/1.2 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
h1 {
  margin: 0;
  max-width: 820px;
  font-size: clamp(42px, 7vw, 76px);
  line-height: 0.96;
  letter-spacing: 0;
}
.page-hero p {
  max-width: 740px;
  margin: 18px 0 0;
  color: var(--muted);
  font-size: 18px;
  line-height: 1.65;
}
.doc-body {
  padding-top: 28px;
}
.doc-body h1 {
  display: none;
}
.doc-body h2 {
  margin: 36px 0 12px;
  font-size: 28px;
  line-height: 1.1;
  letter-spacing: 0;
}
.doc-body p,
.doc-body li {
  color: var(--muted);
  font-size: 16px;
  line-height: 1.7;
}
.doc-body p {
  margin: 0 0 16px;
}
.doc-body ul {
  padding-left: 22px;
}
pre {
  overflow: auto;
  padding: 16px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--code-bg);
}
code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.92em;
}
.feature-grid,
.gallery-grid,
.topic-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
  padding: 28px 0 6px;
}
.feature-grid a,
.gallery-card,
.topic-card {
  display: grid;
  gap: 12px;
  min-height: 178px;
  padding: 18px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--paper);
  box-shadow: var(--shadow);
  text-decoration: none;
}
.feature-grid a:hover,
.gallery-card:hover,
.topic-card:hover {
  border-color: var(--accent);
}
.feature-grid span,
.gallery-card span,
.topic-card span,
.notebook-head span {
  color: var(--accent-2);
  font: 760 12px/1.2 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.feature-grid strong,
.gallery-card strong {
  font-size: 22px;
  line-height: 1.12;
}
.feature-grid p,
.gallery-card p,
.topic-card p {
  margin: 0;
  color: var(--muted);
  line-height: 1.55;
}
.topic-card {
  min-height: 140px;
}
.gallery-card img,
.card-visual {
  width: 100%;
  aspect-ratio: 16 / 9;
  object-fit: cover;
  border-radius: 6px;
  border: 1px solid var(--line);
  background:
    linear-gradient(135deg, rgb(78 123 67 / 0.16), rgb(178 79 59 / 0.12)),
    repeating-linear-gradient(90deg, transparent 0 18px, rgb(17 20 17 / 0.05) 18px 19px);
}
.notebook-panel {
  margin-top: 28px;
  overflow: hidden;
  border: 1px solid var(--line-strong);
  border-radius: 8px;
  background: var(--paper);
  box-shadow: var(--shadow);
}
.example-panel {
  display: grid;
  grid-template-columns: minmax(0, 1.35fr) minmax(280px, 0.65fr);
  gap: 18px;
  margin-top: 28px;
  padding: 16px;
  border: 1px solid var(--line-strong);
  border-radius: 8px;
  background: var(--paper);
  box-shadow: var(--shadow);
}
.example-panel img {
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--code-bg);
}
.example-panel div {
  display: grid;
  align-content: start;
  gap: 12px;
}
.example-panel span {
  color: var(--accent-2);
  font: 760 12px/1.2 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.example-panel strong {
  font-size: 24px;
  line-height: 1.12;
}
.example-panel p {
  margin: 0;
  color: var(--muted);
  line-height: 1.6;
}
.inline-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.inline-actions a {
  padding: 8px 10px;
  border: 1px solid var(--line);
  border-radius: 6px;
  text-decoration: none;
  background: #fff;
}
.notebook-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 14px 16px;
  border-bottom: 1px solid var(--line);
  background: #f7f8f4;
}
.notebook-head div {
  display: grid;
  gap: 4px;
}
.notebook-head strong {
  font-size: 16px;
}
.notebook-head a {
  flex: 0 0 auto;
  padding: 8px 10px;
  border: 1px solid var(--line);
  border-radius: 6px;
  color: var(--ink);
  text-decoration: none;
  background: #fff;
}
iframe {
  display: block;
  width: 100%;
  height: min(900px, 78vh);
  border: 0;
  background: #fff;
}
@media (max-width: 1100px) {
  .layout,
  .layout.wide {
    grid-template-columns: minmax(0, 1fr);
  }
  .sidebar,
  .right-rail {
    display: none;
  }
}
@media (max-width: 780px) {
  .topbar {
    align-items: flex-start;
    flex-direction: column;
    padding: 14px 18px;
  }
  .topbar nav {
    flex-wrap: wrap;
  }
  .layout,
  .layout.wide {
    padding: 24px 18px 56px;
  }
  .feature-grid,
  .gallery-grid,
  .topic-grid {
    grid-template-columns: 1fr;
  }
  .example-panel {
    grid-template-columns: 1fr;
  }
  h1 {
    font-size: 42px;
  }
  iframe {
    height: 680px;
  }
}
"""
