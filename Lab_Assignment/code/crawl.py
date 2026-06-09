import asyncio
import os
from crawl4ai import AsyncWebCrawler

async def main():
    # 1. Điền 5 đường link bài báo của bạn vào danh sách dưới đây
    urls = [
        "https://vnexpress.net/anh-em-ca-si-chi-dan-ru-nhieu-nguoi-choi-ma-tuy-nhu-the-nao-4929804.html",
        "https://vnexpress.net/nguoi-mau-andrea-aybar-va-ca-si-chi-dan-bi-bat-4814295.html",
        "https://vnexpress.net/ca-si-miu-le-bi-bat-voi-cao-buoc-to-chuc-su-dung-ma-tuy-5074769.html",
        "https://vnexpress.net/ca-si-long-nhat-son-ngoc-minh-bi-bat-vi-lien-quan-ma-tuy-5060857.html",
        "https://vnexpress.net/20-nam-hoat-dong-cua-miu-le-truoc-khi-bi-bat-qua-tang-dung-ma-tuy-5072922.html"
    ]

    # Cấu hình đường dẫn lưu file theo yêu cầu: data/landing/news
    output_dir = os.path.join("data", "landing", "news")
    os.makedirs(output_dir, exist_ok=True)

    print("Khởi động quá trình crawl dữ liệu...")
    print(f"Thư mục lưu trữ: {output_dir}\n")

    # 2. Khởi tạo AsyncWebCrawler
    async with AsyncWebCrawler() as crawler:
        for index, url in enumerate(urls):
            # Kiểm tra xem link đã được người dùng nhập chưa (bỏ qua các link mẫu ban đầu)
            if url.startswith("http") and "nhap_link_bai_bao" not in url:
                print(f"[{index + 1}/5] Đang thu thập dữ liệu từ: {url}")
                
                # Thực hiện crawl dữ liệu bài báo
                result = await crawler.arun(url=url)
                
                if result.success:
                    print(f"✅ Thành công!")
                    
                    # Tạo tên file an toàn dựa trên số thứ tự (hoặc bạn có thể tùy chỉnh theo tiêu đề)
                    filename = f"news_output_{index + 1}.txt"
                    filepath = os.path.join(output_dir, filename)
                    
                    # Lưu nội dung bài báo dưới dạng Markdown vào đúng thư mục yêu cầu
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(result.markdown)
                        
                    print(f"💾 Đã lưu output vào: {filepath}\n")
                else:
                    print(f"❌ Thất bại khi crawl: {result.error_message}\n")
            else:
                print(f"[{index + 1}/5] Bỏ qua: Vui lòng nhập URL hợp lệ vào vị trí số {index + 1}\n")

if __name__ == "__main__":
    # Chạy vòng lặp sự kiện bất đồng bộ
    asyncio.run(main())