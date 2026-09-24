"""Capture README screenshots from the running apps (developer tool, needs `pip install playwright`).

Run the apps against the photo-free database first (see scripts/make_screenshot_data.py), then:

    TRACEAI_SHOT_OFFICER_PASSWORD=... TRACEAI_SHOT_ADMIN_PASSWORD=... python -m scripts.capture_screenshots

Uses the system Microsoft Edge/Chrome, so no separate browser download is needed.
"""
import os
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "img"
CONSOLE = os.getenv("TRACEAI_SHOT_CONSOLE", "http://localhost:8511")
PORTAL = os.getenv("TRACEAI_SHOT_PORTAL", "http://localhost:8512")
CHANNEL = os.getenv("TRACEAI_SHOT_BROWSER", "msedge")


def login(page, username: str, password: str) -> None:
    page.goto(CONSOLE)
    page.get_by_role("textbox", name="Username").fill(username)
    page.get_by_role("textbox", name="Password").fill(password)
    page.get_by_role("button", name="Sign in").click()
    page.wait_for_selector("text=Signed in as", timeout=120000)


def settle(page, ms: int = 4500) -> None:
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(ms)  # map tiles load inside an iframe


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=CHANNEL, headless=True)

        # Public portal: appeals plus the tip form.
        page = browser.new_page(viewport={"width": 1100, "height": 1500})
        page.goto(PORTAL)
        page.wait_for_selector("text=Current appeals", timeout=120000)
        settle(page, 1500)
        page.screenshot(path=OUT / "public-portal.png")
        page.close()

        # Officer console: case view, plus frames for a short GIF.
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        login(page, "officer", os.environ["TRACEAI_SHOT_OFFICER_PASSWORD"])
        page.wait_for_selector("text=Leads to review", timeout=120000)
        settle(page)
        frames = [OUT / "_f0.png"]
        page.screenshot(path=frames[0])
        page.screenshot(path=OUT / "officer-console.png")
        summaries = page.locator("details summary")
        for i in range(3):
            if summaries.count() > i:
                summaries.nth(i).click()
                page.wait_for_timeout(900)
                frames.append(OUT / f"_f{i + 1}.png")
                page.screenshot(path=frames[-1])
        page.close()

        # Admin: users and the tamper-evident audit log.
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        login(page, "admin", os.environ["TRACEAI_SHOT_ADMIN_PASSWORD"])
        page.get_by_role("tab", name="Admin").click()
        page.wait_for_selector("text=Audit chain verified", timeout=120000)
        page.wait_for_function("document.querySelectorAll('[data-testid=stDataFrame]').length >= 2", timeout=120000)
        settle(page, 3500)  # the audit table is drawn on a canvas a moment after it mounts
        page.screenshot(path=OUT / "admin-audit.png")
        browser.close()

    imgs = [Image.open(f).convert("RGB") for f in frames]
    w = 960
    imgs = [im.resize((w, int(im.height * w / im.width)), Image.LANCZOS) for im in imgs]
    imgs[0].save(OUT / "officer-flow.gif", save_all=True, append_images=imgs[1:], duration=1800, loop=0, optimize=True)
    for f in frames:
        f.unlink()
    print("Wrote:", *sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
