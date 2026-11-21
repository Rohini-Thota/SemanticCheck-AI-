"""PAN dataset loader scaffold: parses annotation XML and resolves document references."""
from pathlib import Path
import xml.etree.ElementTree as ET

def parse_pan_annotations(xml_path):
    root = ET.parse(xml_path).getroot()
    records = []
    for item in root.iter():
        attrs = item.attrib
        if "source_reference" in attrs or "reused_reference" in attrs:
            records.append({
                "suspicious_reference": attrs.get("suspicious_reference", ""),
                "source_reference": attrs.get("source_reference", attrs.get("reused_reference", "")),
                "reuse_kind": attrs.get("reuse_kind", attrs.get("type", "unknown"))
            })
    return records

def find_document(root_dir, reference):
    root = Path(root_dir)
    matches = list(root.rglob(reference))
    return str(matches[0]) if matches else None
