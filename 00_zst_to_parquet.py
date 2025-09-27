#!/usr/bin/env python3
"""
Convert Reddit .zst files to parquet format for processing.
"""

import pandas as pd
import json
import zstandard as zstd
import io
from pathlib import Path
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def stream_zst_to_dataframe(zst_path, max_rows=None):
    """Stream a .zst file and convert to DataFrame."""
    print(f"Processing {zst_path}...")
    
    data = []
    total_rows = 0
    
    with open(zst_path, 'rb') as fh:
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='ignore', newline='')
            
            for line in tqdm(text_stream, desc=f"Reading {Path(zst_path).name}"):
                if max_rows and total_rows >= max_rows:
                    break
                    
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    obj = json.loads(line)
                    data.append(obj)
                    total_rows += 1
                except Exception as e:
                    continue
    
    print(f"Loaded {len(data)} records from {zst_path}")
    return pd.DataFrame(data)

def clean_dataframe(df):
    """Clean DataFrame for parquet compatibility."""
    if df.empty:
        return df
    
    # Convert complex objects to strings
    for col in df.columns:
        if df[col].dtype == 'object':
            # Convert dict/list objects to strings
            df[col] = df[col].astype(str)
    
    return df

def filter_by_time_and_subreddit(df, config):
    """Filter data by time and subreddit."""
    if df.empty:
        return df
    
    # Convert created_utc to datetime
    if 'created_utc' in df.columns:
        df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s', utc=True)
        
        # Filter by time window
        start_time = pd.to_datetime(config['filters']['start_utc'], utc=True)
        end_time = pd.to_datetime(config['filters']['end_utc'], utc=True)
        
        df = df[(df['created_utc'] >= start_time) & (df['created_utc'] <= end_time)]
        print(f"After time filtering: {len(df)} records")
    
    # Filter by subreddit
    if 'subreddit' in df.columns and config['filters']['subreddits']:
        df = df[df['subreddit'].isin(config['filters']['subreddits'])]
        print(f"After subreddit filtering: {len(df)} records")
    
    # Clean DataFrame
    df = clean_dataframe(df)
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Convert Reddit .zst files to parquet")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--max-rows", type=int, default=None, help="Maximum rows to process (for testing)")
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("=== Converting .zst files to parquet ===")
    
    # Process comments
    comments_path = config['paths']['raw_comments']
    if Path(comments_path).exists():
        print(f"Processing comments: {comments_path}")
        comments_df = stream_zst_to_dataframe(comments_path, args.max_rows)
        comments_df = filter_by_time_and_subreddit(comments_df, config)
        
        # Save comments
        output_path = Path(config['paths']['mini_dir']) / "comments_part000.parquet"
        output_path.parent.mkdir(exist_ok=True)
        comments_df.to_parquet(output_path, index=False)
        print(f"Saved {len(comments_df)} comments to {output_path}")
        
        # Create parts file
        parts_file = output_path.parent / "comments_parts.txt"
        with open(parts_file, 'w') as f:
            f.write("comments_part000.parquet\n")
    else:
        print(f"Comments file not found: {comments_path}")
    
    # Process submissions
    submissions_path = config['paths']['raw_submissions']
    if Path(submissions_path).exists():
        print(f"Processing submissions: {submissions_path}")
        submissions_df = stream_zst_to_dataframe(submissions_path, args.max_rows)
        submissions_df = filter_by_time_and_subreddit(submissions_df, config)
        
        # Save submissions
        output_path = Path(config['paths']['mini_dir']) / "submissions_part000.parquet"
        output_path.parent.mkdir(exist_ok=True)
        submissions_df.to_parquet(output_path, index=False)
        print(f"Saved {len(submissions_df)} submissions to {output_path}")
        
        # Create parts file
        parts_file = output_path.parent / "submissions_parts.txt"
        with open(parts_file, 'w') as f:
            f.write("submissions_part000.parquet\n")
    else:
        print(f"Submissions file not found: {submissions_path}")
    
    print("=== Conversion completed ===")

if __name__ == "__main__":
    main()
