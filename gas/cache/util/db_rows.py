"""
Script to display the first N rows of either the prompts or media table.
"""

import argparse
import sys
import sqlite3
import json
from pathlib import Path
from datetime import datetime


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def format_timestamp(timestamp):
    """Format timestamp as human-readable string."""
    if timestamp is None:
        return "N/A"
    try:
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, OSError):
        return str(timestamp)


def truncate_string(text, max_length=50):
    """Truncate string to max_length with ellipsis if needed."""
    if text is None:
        return "N/A"
    text = str(text)
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def display_prompts_table(db_path: Path, n_rows: int):
    """Display the first N rows of the prompts table."""
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT id, content, content_type, created_at, used_count, last_used, 
                       source_media_id
                FROM prompts 
                ORDER BY created_at DESC 
                LIMIT ?
                """,
                (n_rows,),
            )
            rows = cursor.fetchall()

            if not rows:
                print(f"{Colors.YELLOW}No prompts found in the database.{Colors.END}")
                return

            # Print header
            print(
                f"{Colors.BOLD}{Colors.HEADER}📝 Prompts Table - First {len(rows)} rows{Colors.END}"
            )
            print("=" * 160)

            # First set of columns (basic info)
            print(
                f"{Colors.BOLD}{Colors.CYAN}{'ID':<12} {'Type':<12} {'Content':<60} {'Created':<20} {'Used':<6} {'Last Used':<20}{Colors.END}"
            )
            print("-" * 160)

            # Print basic info for each row
            for row in rows:
                id_short = row["id"][:8] + "..." if row["id"] else "N/A"
                content_preview = truncate_string(row["content"], 55)
                created_str = format_timestamp(row["created_at"])
                last_used_str = format_timestamp(row["last_used"])

                print(
                    f"{id_short:<12} {row['content_type']:<12} {content_preview:<60} {created_str:<20} {row['used_count']:<6} {last_used_str:<20}"
                )

            print("=" * 160)

            # Second set of columns (source info)
            print(
                f"\n{Colors.BOLD}{Colors.CYAN}{'ID':<12} {'Source Media ID':<30} {'Full Content Preview':<80}{Colors.END}"
            )
            print("-" * 160)

            for row in rows:
                id_short = row["id"][:8] + "..." if row["id"] else "N/A"
                media_id_short = (
                    row["source_media_id"][:25] + "..."
                    if row["source_media_id"] and len(row["source_media_id"]) > 25
                    else (row["source_media_id"] or "N/A")
                )
                full_content = truncate_string(row["content"], 75)

                print(f"{id_short:<12} {media_id_short:<30} {full_content:<80}")

            print("=" * 160)

            print(f"\n{Colors.GREEN}Total rows displayed: {len(rows)}{Colors.END}")

    except Exception as e:
        print(f"{Colors.RED}❌ Error reading prompts table: {e}{Colors.END}")


def display_media_table(db_path: Path, n_rows: int, source_type_filter: str = None):
    """Display the first N rows of the media table."""
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Build query with optional source_type filter
            if source_type_filter:
                query = """
                    SELECT id, prompt_id, file_path, modality, media_type, source_type,
                           model_name, download_url, scraper_name, dataset_name, 
                           dataset_source_file, dataset_index, created_at, generation_args,
                           mask_path, timestamp, resolution, file_size, format
                    FROM media 
                    WHERE source_type = ?
                    ORDER BY created_at DESC 
                    LIMIT ?
                """
                cursor = conn.execute(query, (source_type_filter, n_rows))
            else:
                query = """
                    SELECT id, prompt_id, file_path, modality, media_type, source_type,
                           model_name, download_url, scraper_name, dataset_name, 
                           dataset_source_file, dataset_index, created_at, generation_args,
                           mask_path, timestamp, resolution, file_size, format
                    FROM media 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """
                cursor = conn.execute(query, (n_rows,))
            rows = cursor.fetchall()

            if not rows:
                filter_msg = (
                    f" with source_type='{source_type_filter}'"
                    if source_type_filter
                    else ""
                )
                print(
                    f"{Colors.YELLOW}No media found in the database{filter_msg}.{Colors.END}"
                )
                return

            # Print header
            filter_info = (
                f" (filtered by source_type='{source_type_filter}')"
                if source_type_filter
                else ""
            )
            print(
                f"{Colors.BOLD}{Colors.HEADER}🎬 Media Table - First {len(rows)} rows{filter_info}{Colors.END}"
            )
            print("=" * 200)

            # First set of columns (basic info)
            print(
                f"{Colors.BOLD}{Colors.CYAN}{'ID':<12} {'Prompt ID':<12} {'File Path':<35} {'Modality':<10} {'Type':<12} {'Source':<10} {'Created':<20} {'Size':<12}{Colors.END}"
            )
            print("-" * 200)

            # Print basic info for each row
            for row in rows:
                id_short = row["id"][:8] + "..." if row["id"] else "N/A"
                prompt_id_short = (
                    row["prompt_id"][:8] + "..." if row["prompt_id"] else "N/A"
                )
                file_path_short = truncate_string(
                    Path(row["file_path"]).name if row["file_path"] else "N/A", 30
                )
                created_str = format_timestamp(row["created_at"])

                # Format file size
                size_str = "N/A"
                if row["file_size"]:
                    if row["file_size"] < 1024:
                        size_str = f"{row['file_size']}B"
                    elif row["file_size"] < 1024 * 1024:
                        size_str = f"{row['file_size']/1024:.1f}KB"
                    else:
                        size_str = f"{row['file_size']/(1024*1024):.1f}MB"

                print(
                    f"{id_short:<12} {prompt_id_short:<12} {file_path_short:<35} {row['modality'] or 'N/A':<10} {row['media_type'] or 'N/A':<12} {row['source_type'] or 'N/A':<10} {created_str:<20} {size_str:<12}"
                )

            print("=" * 200)

            # Second set of columns (source details)
            print(
                f"\n{Colors.BOLD}{Colors.CYAN}{'ID':<12} {'Model Name':<25} {'Dataset Name':<25} {'Scraper':<15} {'Download URL':<40} {'Format':<8} {'Resolution':<12}{Colors.END}"
            )
            print("-" * 200)

            for row in rows:
                id_short = row["id"][:8] + "..." if row["id"] else "N/A"
                model_name = (
                    truncate_string(row["model_name"], 22)
                    if row["model_name"]
                    else "N/A"
                )
                dataset_name = (
                    truncate_string(row["dataset_name"], 22)
                    if row["dataset_name"]
                    else "N/A"
                )
                scraper_name = (
                    truncate_string(row["scraper_name"], 12)
                    if row["scraper_name"]
                    else "N/A"
                )
                download_url = (
                    truncate_string(row["download_url"], 37)
                    if row["download_url"]
                    else "N/A"
                )
                format_str = row["format"] or "N/A"

                # Parse resolution
                resolution_str = "N/A"
                if row["resolution"]:
                    try:
                        resolution_data = json.loads(row["resolution"])
                        if len(resolution_data) == 2:
                            resolution_str = (
                                f"{resolution_data[0]}x{resolution_data[1]}"
                            )
                    except (json.JSONDecodeError, TypeError, IndexError):
                        resolution_str = str(row["resolution"])[:10]

                print(
                    f"{id_short:<12} {model_name:<25} {dataset_name:<25} {scraper_name:<15} {download_url:<40} {format_str:<8} {resolution_str:<12}"
                )

            print("=" * 200)

            # Third set of columns (additional details)
            print(
                f"\n{Colors.BOLD}{Colors.CYAN}{'ID':<12} {'Dataset File':<30} {'Dataset Index':<15} {'Mask Path':<30} {'Timestamp':<12} {'Has Gen Args':<12}{Colors.END}"
            )
            print("-" * 200)

            for row in rows:
                id_short = row["id"][:8] + "..." if row["id"] else "N/A"
                dataset_file = (
                    truncate_string(row["dataset_source_file"], 27)
                    if row["dataset_source_file"]
                    else "N/A"
                )
                dataset_index = (
                    truncate_string(str(row["dataset_index"]), 12)
                    if row["dataset_index"]
                    else "N/A"
                )
                mask_path = truncate_string(
                    Path(row["mask_path"]).name if row["mask_path"] else "N/A", 27
                )
                timestamp_str = str(row["timestamp"]) if row["timestamp"] else "N/A"
                has_generation_args = "Yes" if row["generation_args"] else "No"

                print(
                    f"{id_short:<12} {dataset_file:<30} {dataset_index:<15} {mask_path:<30} {timestamp_str:<12} {has_generation_args:<12}"
                )

            print("=" * 200)

            # Show example generation_args if any rows have it
            generation_args_examples = []
            for row in rows:
                if row["generation_args"]:
                    try:
                        generation_args_dict = json.loads(row["generation_args"])
                        generation_args_examples.append(
                            (row["id"][:8] + "...", generation_args_dict)
                        )
                    except json.JSONDecodeError:
                        generation_args_examples.append(
                            (row["id"][:8] + "...", row["generation_args"])
                        )

            if generation_args_examples:
                print(
                    f"\n{Colors.BOLD}{Colors.YELLOW}🔧 Example Generation Args (JSON){Colors.END}"
                )
                print("-" * 80)
                for row_id, generation_args in generation_args_examples[
                    :3
                ]:  # Show up to 3 examples
                    print(f"\n{Colors.CYAN}Row ID: {row_id}{Colors.END}")
                    if isinstance(generation_args, dict):
                        print(json.dumps(generation_args, indent=2, ensure_ascii=False))
                    else:
                        print(str(generation_args))
                    print("-" * 40)

            print(f"\n{Colors.GREEN}Total rows displayed: {len(rows)}{Colors.END}")

    except Exception as e:
        print(f"{Colors.RED}❌ Error reading media table: {e}{Colors.END}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Display first N rows of prompts or media table"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="~/.cache/sn34/prompts.db",
        help="Path to the prompt database",
    )
    parser.add_argument(
        "--table",
        type=str,
        choices=["prompts", "media"],
        required=True,
        help="Table to display (prompts or media)",
    )
    parser.add_argument(
        "--rows", type=int, default=10, help="Number of rows to display (default: 10)"
    )
    parser.add_argument(
        "--source-type",
        type=str,
        choices=["scraper", "dataset", "generated"],
        help="Filter media table by source type (only applies to media table)",
    )

    args = parser.parse_args()

    # Validate source-type is only used with media table
    if args.source_type and args.table != "media":
        print(
            f"{Colors.RED}❌ --source-type can only be used with --table media{Colors.END}"
        )
        sys.exit(1)

    db_path = Path(args.db_path).expanduser()

    if not db_path.exists():
        print(f"{Colors.RED}❌ Database not found: {db_path}{Colors.END}")
        print(
            "💡 The database will be created automatically when prompts are first added."
        )
        sys.exit(1)

    print(f"{Colors.CYAN}📍 Database: {db_path}{Colors.END}")
    filter_info = f" (source_type: {args.source_type})" if args.source_type else ""
    print(
        f"{Colors.CYAN}📊 Displaying first {args.rows} rows from {args.table} table{filter_info}{Colors.END}"
    )
    print()

    if args.table == "prompts":
        display_prompts_table(db_path, args.rows)
    elif args.table == "media":
        display_media_table(db_path, args.rows, args.source_type)

    print(f"\n{Colors.GREEN}✅ Table display completed!{Colors.END}")


if __name__ == "__main__":
    main()
