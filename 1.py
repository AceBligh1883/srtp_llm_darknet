import os

def extract_onion_from_filename(filename):
    # 分割文件名并提取第二个和第三个下划线之间的部分
    parts = filename.split('_')
    if len(parts) >= 3:
        return parts[2]  # 第二个和第三个下划线之间的部分
    return None

def count_unique_onion_urls(folder_path):
    unique_urls = set()  # 使用set来去重
    for filename in os.listdir(folder_path):
        onion = extract_onion_from_filename(filename)
        if onion:
            unique_urls.add(onion)  # 添加到set中，自动去重
    return len(unique_urls)

# 设定文件夹路径
folder_path = 'data/input/files'  # 替换成你的文件夹路径
unique_count = count_unique_onion_urls(folder_path)

print(f"Unique .onion URLs found: {unique_count}")
