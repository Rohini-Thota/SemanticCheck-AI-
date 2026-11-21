
import re
import csv
import random
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

def read_text(path):
    for encoding in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return ""

def build_pan_dataset(corpus):
    corpus = Path(corpus)
    source_dir = corpus / "source-document"
    suspicious_dir = corpus / "suspicious-document"

    if not source_dir.exists() or not suspicious_dir.exists():
        raise FileNotFoundError(
            "Expected source-document and suspicious-document folders."
        )

    sources = {p.name.lower(): p for p in source_dir.rglob("*.txt")}
    suspicious = {p.name.lower(): p for p in suspicious_dir.rglob("*.txt")}

    # Also index documents by filename without extension.
    source_stems = {p.stem.lower(): p for p in sources.values()}

    pairs = []
    referenced_sources = {}

    for xml_file in suspicious_dir.rglob("*.xml"):
        try:
            root = ET.parse(xml_file).getroot()
        except ET.ParseError:
            continue

        suspicious_ref = root.attrib.get(
            "reference", xml_file.stem + ".txt"
        ).replace("\\", "/").split("/")[-1].lower()

        suspicious_path = suspicious.get(suspicious_ref)
        if suspicious_path is None:
            suspicious_path = next(
                (p for p in suspicious.values()
                 if p.stem.lower() == Path(suspicious_ref).stem.lower()),
                None
            )

        if suspicious_path is None:
            continue

        suspicious_text = read_text(suspicious_path)

        for element in root.iter():
            attrs = {k.lower(): v for k, v in element.attrib.items()}

            source_ref = attrs.get("source_reference")
            if not source_ref:
                continue

            source_name = source_ref.replace("\\", "/").split("/")[-1].lower()
            source_path = sources.get(source_name)

            if source_path is None:
                source_path = source_stems.get(Path(source_name).stem.lower())

            if source_path is None:
                continue

            try:
                this_offset = int(attrs["this_offset"])
                this_length = int(attrs["this_length"])
                source_offset = int(attrs["source_offset"])
                source_length = int(attrs["source_length"])
            except (KeyError, ValueError):
                continue

            suspicious_segment = suspicious_text[
                this_offset:this_offset + this_length
            ]
            source_text = read_text(source_path)
            source_segment = source_text[
                source_offset:source_offset + source_length
            ]

            if len(suspicious_segment.strip()) < 30:
                continue
            if len(source_segment.strip()) < 30:
                continue

            pairs.append({
                "sentence1": suspicious_segment.strip(),
                "sentence2": source_segment.strip(),
                "label": 1
            })

            referenced_sources.setdefault(
                suspicious_path.name, set()
            ).add(source_path.name)

    # Create negative pairs from unrelated source documents.
    random.seed(42)
    source_paths = list(sources.values())

    for suspicious_path in suspicious.values():
        suspicious_text = read_text(suspicious_path)
        sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", suspicious_text)
            if len(s.strip()) >= 30
        ]

        if not sentences:
            continue

        used = referenced_sources.get(suspicious_path.name, set())
        unrelated = [p for p in source_paths if p.name not in used]

        if not unrelated:
            continue

        random.shuffle(sentences)
        random.shuffle(unrelated)

        count = min(len(sentences), len(unrelated), 2)
        for i in range(count):
            source_text = read_text(unrelated[i])
            source_sentences = [
                s.strip() for s in re.split(r"(?<=[.!?])\s+", source_text)
                if len(s.strip()) >= 30
            ]
            if not source_sentences:
                continue

            pairs.append({
                "sentence1": sentences[i],
                "sentence2": random.choice(source_sentences),
                "label": 0
            })

    output = Path("training/pan_pairs.csv")
    output.parent.mkdir(exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sentence1", "sentence2", "label"]
        )
        writer.writeheader()
        writer.writerows(pairs)

    positive = sum(p["label"] == 1 for p in pairs)
    negative = sum(p["label"] == 0 for p in pairs)

    print("PAN dataset preparation complete!")
    print("Positive annotated pairs:", positive)
    print("Negative sampled pairs:", negative)
    print("Total pairs:", len(pairs))
    print("Saved to:", output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    args = parser.parse_args()
    build_pan_dataset(args.corpus)
