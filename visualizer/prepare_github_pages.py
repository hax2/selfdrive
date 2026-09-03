#!/usr/bin/env python3
"""
prepare_github_pages.py
Prepares a self-contained, 100% static GitHub Pages deployment in the /docs directory.
Copies curated sample triptychs, overlays, raw RGBs, thesis figures, and qualitative images,
and updates all data URLs to relative paths so it loads flawlessly on:
https://<username>.github.io/<repo>/ and locally without any server dependencies.
"""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

PRESET_SAMPLE_IDS = [
    "mixed_pln_1278", "mixed_425", "mixed_459", "mixed_pln_961", "mixed_89",
    "mixed_pln_1188", "mixed_100", "mixed_pln_610", "mixed_pln_546", "mixed_306",
    "mixed_394", "mixed_313", "mixed_pln_1209", "mixed_pln_119", "mixed_pln_1033",
    "mixed_pln_1282", "mixed_pln_1306", "mixed_104", "mixed_106", "mixed_108",
    "mixed_113", "mixed_115", "mixed_118", "mixed_123"
]

def main():
    print("Preparing self-contained GitHub Pages deployment in docs/...")
    
    # 1. Clean & recreate docs structure
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "data").mkdir(parents=True, exist_ok=True)
    (DOCS / "media" / "look_here").mkdir(parents=True, exist_ok=True)
    (DOCS / "media" / "qualitative").mkdir(parents=True, exist_ok=True)
    (DOCS / "media" / "samples" / "triptychs").mkdir(parents=True, exist_ok=True)
    (DOCS / "media" / "samples" / "overlays").mkdir(parents=True, exist_ok=True)
    (DOCS / "media" / "samples" / "images").mkdir(parents=True, exist_ok=True)
    (DOCS / "media" / "samples" / "masks").mkdir(parents=True, exist_ok=True)

    # 2. Touch .nojekyll (bypasses Jekyll on GitHub Pages)
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")

    # 3. Copy Thesis Figures from 'look here'
    look_here_src = ROOT / "look here"
    for f in look_here_src.glob("*.png"):
        shutil.copy2(f, DOCS / "media" / "look_here" / f.name)
    print(f"Copied {len(list(look_here_src.glob('*.png')))} thesis figures to docs/media/look_here/")

    # 4. Copy Qualitative Figures from reports/figures/qualitative
    qual_src = ROOT / "reports" / "figures" / "qualitative"
    for f in qual_src.glob("*.png"):
        shutil.copy2(f, DOCS / "media" / "qualitative" / f.name)
    print(f"Copied {len(list(qual_src.glob('*.png')))} qualitative images to docs/media/qualitative/")

    # 5. Extract pixel-aligned 640x384 crops from triptychs
    from PIL import Image
    copied_samples = []
    for sid in PRESET_SAMPLE_IDS:
        t_src = ROOT / "outputs" / "mixed_binary_traversability" / "test_review" / "all_triptychs" / f"{sid}.png"
        if t_src.exists():
            shutil.copy2(t_src, DOCS / "media" / "samples" / "triptychs" / f"{sid}.png")
            im = Image.open(t_src)
            if im.size == (1920, 384):
                rgb = im.crop((0, 0, 640, 384))
                gt = im.crop((640, 0, 1280, 384))
                pred = im.crop((1280, 0, 1920, 384))
                rgb.save(DOCS / "media" / "samples" / "images" / f"{sid}.png")
                gt.save(DOCS / "media" / "samples" / "masks" / f"{sid}.png")
                pred.save(DOCS / "media" / "samples" / "overlays" / f"{sid}.png")
            copied_samples.append(sid)
    print(f"Extracted {len(copied_samples)} pixel-aligned 640x384 sample sets into docs/media/samples/")

    # 6. Load benchmark data and update all URLs to relative paths
    src_data_file = ROOT / "visualizer" / "public" / "data" / "benchmark_data.json"
    data = json.loads(src_data_file.read_text(encoding="utf-8"))

    # Update thesis figures URLs
    for tf in data.get("thesis_figures", []):
        name = Path(tf["img_url"]).name
        tf["img_url"] = f"./media/look_here/{name}"

    # Update qualitative URLs
    for cat_name, items in data.get("qualitative", {}).items():
        if isinstance(items, list):
            for it in items:
                for key in ["triptych_url", "overlay_url", "detail_img", "img_url"]:
                    if key in it and it[key]:
                        fn = Path(it[key]).name
                        # Check where fn exists
                        if (DOCS / "media" / "samples" / "triptychs" / fn).exists():
                            it[key] = f"./media/samples/triptychs/{fn}"
                        elif (DOCS / "media" / "samples" / "overlays" / fn).exists():
                            it[key] = f"./media/samples/overlays/{fn}"
                        elif (DOCS / "media" / "qualitative" / fn).exists():
                            it[key] = f"./media/qualitative/{fn}"
                        elif (DOCS / "media" / "look_here" / fn).exists():
                            it[key] = f"./media/look_here/{fn}"

    # Keep only samples whose assets are actually packaged. Substituting a
    # representative image for missing samples makes the browser misleading.
    packaged_test_samples = []
    for ts in data.get("test_samples", []):
        sid = ts.get("id", "")
        if sid in copied_samples:
            ts["triptych_url"] = f"./media/samples/triptychs/{sid}.png"
            ts["overlay_url"] = f"./media/samples/overlays/{sid}.png"
            ts["raw_rgb_url"] = f"./media/samples/images/{sid}.png"
            ts["raw_mask_url"] = f"./media/samples/masks/{sid}.png"
            packaged_test_samples.append(ts)
    data["test_samples"] = packaged_test_samples

    # Write relative benchmark_data.json
    (DOCS / "data" / "benchmark_data.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote relative data to docs/data/benchmark_data.json ({ (DOCS / 'data' / 'benchmark_data.json').stat().st_size / 1024:.1f} KB)")

    # 7. Copy HTML, CSS, JS and ensure relative references
    html = (ROOT / "visualizer" / "public" / "index.html").read_text(encoding="utf-8")
    # Replace any absolute paths with relative
    html = html.replace('href="index.css"', 'href="./index.css"')
    html = html.replace('src="app.js"', 'src="./app.js"')
    # Replace default split-img src with relative path
    html = html.replace('/media/outputs/mixed_binary_traversability/test_review/all_overlays/mixed_pln_1278.png', './media/samples/overlays/mixed_pln_1278.png')
    html = html.replace('/media/data_processed_blue_green/test/images/mixed_pln_1278.png', './media/samples/images/mixed_pln_1278.png')
    (DOCS / "index.html").write_text(html, encoding="utf-8")

    css = (ROOT / "visualizer" / "public" / "index.css").read_text(encoding="utf-8")
    (DOCS / "index.css").write_text(css, encoding="utf-8")

    js = (ROOT / "visualizer" / "public" / "app.js").read_text(encoding="utf-8")
    # In JS: fetch('/api/data') -> fetch('./data/benchmark_data.json')
    js = js.replace("fetch('/api/data')", "fetch('./data/benchmark_data.json')")
    # In JS: sample URLs to relative paths
    js = js.replace("const rgbUrl = `/media/data_processed_blue_green/test/images/${sampleId}.png`;", "const rgbUrl = `./media/samples/images/${sampleId}.png`;")
    js = js.replace("const maskUrl = `/media/data_processed_blue_green/test/masks/${sampleId}.png`;", "const maskUrl = `./media/samples/masks/${sampleId}.png`;")
    js = js.replace("const overlayUrl = `/media/outputs/mixed_binary_traversability/test_review/all_overlays/${sampleId}.png`;", "const overlayUrl = `./media/samples/overlays/${sampleId}.png`;")
    js = js.replace("const triptychUrl = `/media/outputs/mixed_binary_traversability/test_review/all_triptychs/${sampleId}.png`;", "const triptychUrl = `./media/samples/triptychs/${sampleId}.png`;")

    (DOCS / "app.js").write_text(js, encoding="utf-8")
    og_image = ROOT / "visualizer" / "public" / "og.png"
    if og_image.exists():
        shutil.copy2(og_image, DOCS / "og.png")
    print("Static files (index.html, index.css, app.js) successfully prepared in docs/!")

if __name__ == "__main__":
    main()
