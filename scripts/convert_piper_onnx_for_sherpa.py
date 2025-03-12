#!/usr/bin/env python3
# Script to convert Piper ONNX models for use with Sherpa-ONNX

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import onnx


def add_meta_data(filename: str, meta_data: Dict[str, Any]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)
    
    # Check if metadata already exists
    has_metadata = False
    for meta in model.metadata_props:
        if meta.key == "model_type" and meta.value == "vits":
            has_metadata = True
            break
    
    if has_metadata:
        print(f"Model {filename} already has Sherpa metadata, skipping")
        return False
    
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)
    return True


def load_config(config_path):
    with open(config_path, "r") as file:
        config = json.load(file)
    return config


def generate_tokens(config, output_path):
    id_map = config["phoneme_id_map"]
    with open(output_path, "w", encoding="utf-8") as f:
        for s, i in id_map.items():
            f.write(f"{s} {i[0]}\n")
    print(f"Generated tokens file: {output_path}")


def process_model(onnx_path: Path, json_path: Path) -> bool:
    """Process a single Piper model for Sherpa compatibility
    
    Args:
        onnx_path: Path to the ONNX model file
        json_path: Path to the JSON config file
        
    Returns:
        True if model was processed, False if skipped
    """
    print(f"\nProcessing model: {onnx_path.name}")
    
    # Load config
    config = load_config(json_path)
    
    # Generate tokens file
    tokens_path = onnx_path.parent / f"{onnx_path.stem}.tokens.txt"
    generate_tokens(config, tokens_path)
    
    # Create metadata
    meta_data = {
        "model_type": "vits",
        "comment": "piper",  # must be piper for models from piper
        "language": config["language"]["name_english"],
        "voice": config["espeak"]["voice"],  # e.g., en-us
        "has_espeak": 1,
        "n_speakers": config["num_speakers"],
        "sample_rate": config["audio"]["sample_rate"],
    }
    
    print("Adding model metadata:")
    for key, value in meta_data.items():
        print(f"  {key}: {value}")
    
    # Add metadata to model
    return add_meta_data(str(onnx_path), meta_data)


def process_directory(directory: Path):
    """Process all Piper models in a directory
    
    Args:
        directory: Path to directory containing Piper models
    """
    print(f"Processing Piper models in directory: {directory}")
    
    # Find all ONNX files in the directory
    onnx_files = list(directory.glob("*.onnx"))
    if not onnx_files:
        print(f"No ONNX files found in {directory}")
        return
    
    print(f"Found {len(onnx_files)} ONNX files")
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for onnx_file in onnx_files:
        # Look for corresponding JSON file
        json_file = Path(f"{onnx_file}.json")
        
        if not json_file.exists():
            print(f"Warning: No JSON config found for {onnx_file.name}, skipping")
            skipped_count += 1
            continue
        
        try:
            # Process the model
            if process_model(onnx_file, json_file):
                processed_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            print(f"Error processing {onnx_file.name}: {e}")
            error_count += 1
    
    print(f"\nProcess completed: {processed_count} models processed, {skipped_count} skipped, {error_count} errors")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Piper ONNX models for use with Sherpa-ONNX"
    )
    parser.add_argument(
        "--dir", type=str, required=True, help="Directory containing Piper models"
    )
    args = parser.parse_args()
    
    directory = Path(args.dir)
    if not directory.exists() or not directory.is_dir():
        print(f"Error: Directory {args.dir} does not exist")
        return
    
    process_directory(directory)


if __name__ == "__main__":
    main()