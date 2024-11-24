

def extract_images_from_parquet(
    parquet_path: Path,
    num_images: int,
    columns: Optional[List[str]] = None,
    seed: Optional[int] = None
) -> List[Tuple[Image.Image, Dict[str, any]]]:
    """
    Extract random images and their metadata from a parquet file.
    
    Args:
        parquet_path: Path to the parquet file
        num_images: Number of images to extract
        columns: Specific columns to include in metadata
        seed: Random seed for sampling
        
    Returns:
        List of tuples containing (PIL Image, metadata dict)
    """
    # Read parquet file
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    
    # Sample random rows
    sample_df = df.sample(n=min(num_images, len(df)), random_state=seed)
    
    results = []
    for _, row in sample_df.iterrows():
        try:
            # Extract image data
            if 'image_bytes' in row:
                img_data = row['image_bytes']
            elif 'image_base64' in row:
                import base64
                img_data = base64.b64decode(row['image_base64'])
            else:
                raise ValueError("No image data column found")
            
            # Convert to PIL Image
            img = Image.open(BytesIO(img_data))
            
            # Extract metadata
            metadata = {}
            if columns:
                for col in columns:
                    if col in row and col not in ['image_bytes', 'image_base64']:
                        metadata[col] = row[col]
            
            results.append((img, metadata))
            
        except Exception as e:
            warnings.warn(f"Failed to extract image: {e}")
            continue
            
    return results