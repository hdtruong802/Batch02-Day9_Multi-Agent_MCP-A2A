"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://vnexpress.net/anh-em-ca-si-chi-dan-ru-nhieu-nguoi-choi-ma-tuy-nhu-the-nao-4929804.html",
    "https://vnexpress.net/nguoi-mau-andrea-aybar-va-ca-si-chi-dan-bi-bat-4814295.html",
    "https://vnexpress.net/ca-si-miu-le-bi-bat-voi-cao-buoc-to-chuc-su-dung-ma-tuy-5074769.html",
    "https://vnexpress.net/ca-si-long-nhat-son-ngoc-minh-bi-bat-vi-lien-quan-ma-tuy-5060857.html",
    "https://vnexpress.net/20-nam-hoat-dong-cua-miu-le-truoc-khi-bi-bat-qua-tang-dung-ma-tuy-5072922.html",
]


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _extract_title(content: str) -> str:
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line.lstrip("# ").strip()
        if line.startswith("## "):
            return line.lstrip("# ").strip()
    return "Unknown"


async def crawl_article(url: str) -> dict:
    """Crawl một bài báo và trả về dict chứa metadata + content."""
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        title = "Unknown"
        if result.success and result.metadata:
            title = result.metadata.get("title") or _extract_title(result.markdown or "")
        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": result.markdown if result.success else "",
        }


def export_txt_to_json():
    """Chuyển file .txt đã crawl sang .json (giữ metadata url cho test suite)."""
    setup_directory()
    for i, url in enumerate(ARTICLE_URLS, 1):
        txt_path = DATA_DIR / f"news_output_{i}.txt"
        json_path = DATA_DIR / f"news_output_{i}.json"
        if not txt_path.exists():
            continue
        content = txt_path.read_text(encoding="utf-8")
        article = {
            "url": url,
            "title": _extract_title(content),
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": content,
        }
        json_path.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [OK] Exported: {json_path.name}")


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)
        filepath = DATA_DIR / f"news_output_{i}.json"
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [OK] Saved: {filepath}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "export":
        print("Export TXT to JSON...")
        export_txt_to_json()
    elif not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
    else:
        asyncio.run(crawl_all())
