import os
from markitdown import MarkItDown

def process_files():
    # 1. Định nghĩa các đường dẫn
    input_dirs = [
        os.path.join("data", "landing", "legal"),
        os.path.join("data", "landing", "news")
    ]
    output_dir = os.path.join("data", "standardized")

    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)

    # 2. Khởi tạo công cụ MarkItDown
    md = MarkItDown()

    print(f"Bắt đầu chuyển đổi dữ liệu, lưu kết quả tại: {output_dir}\n")

    # 3. Lặp qua từng thư mục input
    for directory in input_dirs:
        if not os.path.exists(directory):
            print(f"⚠️ Thư mục không tồn tại (sẽ bỏ qua): {directory}")
            continue

        print(f"📂 Đang quét thư mục: {directory}")
        
        # Lặp qua các file trong thư mục
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            # Chỉ xử lý nếu là file
            if os.path.isfile(file_path):
                try:
                    # Chuyển đổi file
                    result = md.convert(file_path)
                    
                    # Tạo tên file mới với đuôi .md
                    base_name = os.path.splitext(filename)[0]
                    output_filename = f"{base_name}.md"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # Tránh ghi đè nếu có 2 file trùng tên ở 2 thư mục khác nhau
                    counter = 1
                    while os.path.exists(output_path):
                        output_filename = f"{base_name}_{counter}.md"
                        output_path = os.path.join(output_dir, output_filename)
                        counter += 1
                    
                    # Ghi kết quả ra file mới
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(result.text_content)
                        
                    print(f"  ✅ Thành công: {filename} -> {output_filename}")
                except Exception as e:
                    print(f"  ❌ Lỗi khi xử lý {filename}: {e}")
        print("-" * 40)
        
    print("\n🎉 Đã hoàn tất quá trình chuyển đổi!")

if __name__ == "__main__":
    process_files()