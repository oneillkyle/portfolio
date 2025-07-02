# e.g. load from a directory of .txt files
import glob
passages = []
for path in glob.glob("my_docs/**/*.txt", recursive=True):
    with open(path, encoding="utf-8") as f:
        for para in f.read().split("\n\n"):             # split on blank line
            if len(para) > 50:                         # filter very short
                passages.append(para.strip())
