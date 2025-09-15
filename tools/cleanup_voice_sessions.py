#!/usr/bin/env python3
"""
Manual session cleanup utility for BeautyAI voice sessions.

This script allows you to manually clean up old session files when needed.
"""
import argparse
import time
from pathlib import Path
import sys

def cleanup_old_sessions(session_dir: Path, max_age_hours: float = 24.0, dry_run: bool = False):
    """
    Clean up session files older than specified age.
    
    Args:
        session_dir: Directory containing session files
        max_age_hours: Maximum age in hours before deletion
        dry_run: If True, only show what would be deleted without actually deleting
    """
    if not session_dir.exists():
        print(f"❌ Session directory not found: {session_dir}")
        return 0
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    old_files = []
    total_files = 0
    
    for session_file in session_dir.glob("*.json"):
        total_files += 1
        try:
            file_age = current_time - session_file.stat().st_mtime
            if file_age > max_age_seconds:
                old_files.append((session_file, file_age))
        except Exception as e:
            print(f"⚠️ Could not check file {session_file}: {e}")
    
    print(f"📁 Found {total_files} total session files")
    print(f"🕒 Found {len(old_files)} files older than {max_age_hours} hours")
    
    if not old_files:
        print("✅ No old files to clean up")
        return 0
    
    if dry_run:
        print("🔍 DRY RUN - Files that would be deleted:")
        for session_file, age in old_files:
            age_hours = age / 3600
            print(f"  - {session_file.name} (age: {age_hours:.1f} hours)")
        return len(old_files)
    
    # Actually delete the files
    deleted_count = 0
    for session_file, age in old_files:
        try:
            session_file.unlink()
            age_hours = age / 3600
            print(f"🗑️ Deleted: {session_file.name} (age: {age_hours:.1f} hours)")
            deleted_count += 1
        except Exception as e:
            print(f"❌ Failed to delete {session_file}: {e}")
    
    print(f"✅ Successfully deleted {deleted_count} old session files")
    return deleted_count

def main():
    parser = argparse.ArgumentParser(
        description="Clean up old BeautyAI voice session files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Clean files older than 24 hours
  %(prog)s --max-age 12             # Clean files older than 12 hours  
  %(prog)s --max-age 168            # Clean files older than 1 week
  %(prog)s --dry-run                # Show what would be deleted
  %(prog)s --session-dir /custom/path # Use custom session directory
        """
    )
    
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=Path("backend/sessions/voice"),
        help="Session directory path (default: backend/sessions/voice)"
    )
    
    parser.add_argument(
        "--max-age",
        type=float,
        default=24.0,
        help="Maximum age in hours before deletion (default: 24.0)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    
    args = parser.parse_args()
    
    print(f"🧹 BeautyAI Session Cleanup Utility")
    print(f"📂 Session directory: {args.session_dir.absolute()}")
    print(f"⏰ Max age: {args.max_age} hours")
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be deleted")
    print("-" * 50)
    
    try:
        deleted_count = cleanup_old_sessions(
            session_dir=args.session_dir,
            max_age_hours=args.max_age,
            dry_run=args.dry_run
        )
        
        if deleted_count > 0:
            sys.exit(0)  # Success
        else:
            sys.exit(0)  # No files to delete (also success)
            
    except KeyboardInterrupt:
        print("\n🛑 Cleanup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()