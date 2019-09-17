import json
import os


if __name__ == "__main__":
    f_write = open("/Users/hedongfeng/Desktop/GPT2-Chinese/news-keywords/train.json", "w")
    for root, dirs, files in os.walk(top="/Users/hedongfeng/Desktop/GPT2-Chinese/news-keywords"):
        for file in files:
            if file == "train.json": continue
            file_path = os.path.join(root, file)
            with open(file_path, "r") as f:
                for line in f:
                    if len(line) > 0:
                        f_write.write(line)
