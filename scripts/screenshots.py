"""Capture each dashboard tab. Requires the app running on :8517."""
import sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright
OUT = Path(__file__).resolve().parents[1] / "artifacts/screenshots"; OUT.mkdir(parents=True, exist_ok=True)
TABS = [("Trends", 1500), ("Sentiment", 1750), ("Topics", 1150), ("Keywords", 2100), ("Cases correlation", 1500), ("Try the model", 1450)]
with sync_playwright() as p:
    b = p.chromium.launch()
    for region in ["Global", "Mumbai"]:
        for i, (name, h) in enumerate(TABS):
            if region == "Mumbai" and name not in ("Trends", "Sentiment"):
                continue
            pg = b.new_page(viewport={"width": 1440, "height": h}, device_scale_factor=1)
            pg.goto("http://localhost:8517", wait_until="networkidle")
            pg.wait_for_selector("text=Tweets (filtered)", timeout=120000)
            if region != "Global":
                pg.get_by_text(region, exact=True).first.click(); time.sleep(4)
            pg.get_by_role("tab").nth(i).click()
            time.sleep(6)
            errs = pg.locator("[data-testid=stException]").count()
            f = OUT / f"{i+1:02d}-{name.lower().replace(' ', '-')}{'-' + region.lower() if region != 'Global' else ''}.png"
            pg.screenshot(path=str(f))
            print(f, "exceptions:", errs)
            pg.close()
    b.close()
