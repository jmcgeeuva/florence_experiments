import xml.etree.ElementTree as ET
from collections import defaultdict
import csv
import argparse

def format_label(tier_id):
    words = tier_id.strip().split()
    if not words:
        return ""
    return words[0].capitalize() + " " + " ".join(w.lower() for w in words[1:])


def parse_eaf_annotations(eaf_file):
    tree = ET.parse(eaf_file)
    root = tree.getroot()

    # Extract time slots
    time_order = root.find("TIME_ORDER")
    time_slots = {}
    for ts in time_order.findall("TIME_SLOT"):
        ts_id = ts.attrib["TIME_SLOT_ID"]
        ts_value = int(ts.attrib["TIME_VALUE"])
        time_slots[ts_id] = ts_value

    # Extract annotations
    annotations_per_frame = defaultdict(list)
    for tier in root.findall("TIER"):
        if "PARENT_REF" not in tier.attrib:
            continue  # Skip if there's no parent

        tier_id = tier.attrib.get("TIER_ID", "unknown")
        formatted_tier_id = format_label(tier_id)

        for annotation in tier.findall(".//ALIGNABLE_ANNOTATION"):
            start_ts = annotation.attrib["TIME_SLOT_REF1"]
            end_ts = annotation.attrib["TIME_SLOT_REF2"]
            start_time = time_slots[start_ts]
            end_time = time_slots[end_ts]
            value = annotation.find("ANNOTATION_VALUE").text or ""
            if value.strip():  # only if the annotation is non-empty
                for time in range(start_time, end_time + 1):
                    if formatted_tier_id is not []:
                        annotations_per_frame[time].append(formatted_tier_id)

    return annotations_per_frame

def get_eaf_labels(eaf_file_path, frame_times):
    eaf_dict = parse_eaf_annotations(eaf_file_path)

    # Get annotations for a specific frame time in milliseconds
    if frame_times is not None:
        eaf_dict = {}
        for frame_time in frame_times:
            labels_at_frame = frame_annotations.get(frame_time, [])
            eaf_dict[frame_time] = labels_at_frame
    return eaf_dict

def print_eaf(eaf_dict):
    with open("eaf_labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_number", "caption_result"])
        for frame_time, labels_at_frame in eaf_dict.items():
            print(f"Labels at frame {frame_time}ms:", labels_at_frame)
            writer.writerow([frame_time, labels_at_frame])
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--video", default="/standard/spencerNSF/NeuralNetworksProjectVideos/314hours/Video Annotations_314 Hours/110.006.2018_ELA2_Year2_20180205.eaf", help="The name of the user to greet.")
    parser.add_argument('--nargs', nargs='+', default=[3000, 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500])
    args = parser.parse_args()

    out_dict = get_eaf_labels(args.video, args.nargs)
    print_eaf(out_dict)
