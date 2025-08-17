#!/usr/bin/env bash
# Enhanced journal management script for beautyai-api.service
# Usage: ./tools/refresh_api_journal.sh [COMMAND] [OPTIONS]
#
# Commands:
#   clean      - Clean journal history (default: 1 second retention)
#   save       - Save current journal to log file
#   refresh    - Clean history then save fresh journal (default)
#   follow     - Save journal and follow new entries
#   clean-all  - Attempt to remove ALL journal history (requires confirmation or --yes)
#
# Options:
#   -l, --lines N       Number of lines to capture (default: 500)
#   -r, --retain TIME   Retention time for cleaning (default: 1s)
#   -f, --file PATH     Output file path (default: reports/logs/beautyai_api_journal.log)
#   -s, --service NAME  Service name (default: beautyai-api.service)
#   -v, --verbose       Enable verbose output
#   -y, --yes           Non-interactive yes to destructive prompts
#   -g, --global        EXTREME: global journal purge (all services) when combined with clean-all
#   -h, --help          Show this help message
#
# Examples:
#   ./refresh_api_journal.sh clean --retain=1h
#   ./refresh_api_journal.sh save --lines=1000
#   ./refresh_api_journal.sh follow --file=/tmp/api.log

set -euo pipefail

# Default configuration
readonly SCRIPT_NAME="$(basename "$0")"
readonly DEFAULT_SERVICE="beautyai-api.service"
readonly DEFAULT_LOG_PATH="reports/logs/beautyai_api_journal.log"
readonly DEFAULT_LINES=500
readonly DEFAULT_RETAIN="1s"

# Configuration variables
SERVICE="${DEFAULT_SERVICE}"
LOG_PATH="${DEFAULT_LOG_PATH}"
LINES="${DEFAULT_LINES}"
RETAIN="${DEFAULT_RETAIN}"
VERBOSE=0
COMMAND=""
ASSUME_YES=0
GLOBAL_NUKE=0

# Logging functions
log_info() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $*"; }
log_error() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*"; }
log_verbose() { [[ $VERBOSE -eq 1 ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: $*" || true; }

# Help function
show_help() {
    sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
    exit 0
}

# Validation functions
validate_service() {
    if ! systemctl list-unit-files "$SERVICE" &>/dev/null; then
        log_error "Service '$SERVICE' not found"
        return 1
    fi
}

validate_retention() {
    if ! [[ "$RETAIN" =~ ^[0-9]+(s|m|h|d|w|M|y)$ ]]; then
        log_error "Invalid retention format: '$RETAIN'. Use format like: 1s, 5m, 2h, 1d, etc."
        return 1
    fi
}

validate_lines() {
    if ! [[ "$LINES" =~ ^[0-9]+$ ]] || [[ "$LINES" -le 0 ]]; then
        log_error "Lines must be a positive integer, got: '$LINES'"
        return 1
    fi
}

# Check if running as root or with sudo
check_privileges() {
    if [[ $EUID -ne 0 ]] && ! sudo -n true 2>/dev/null; then
        log_error "This script requires sudo privileges for journalctl operations"
        log_error "Either run as root or ensure passwordless sudo is configured"
        return 1
    fi
}

# Journal operations
clean_journal() {
    log_info "Cleaning journal for service: $SERVICE (retention: $RETAIN)"
    
    local journal_size_before
    journal_size_before=$(get_journal_size)
    
    log_verbose "Journal size before cleaning: $journal_size_before"
    
    # Rotate logs first
    if sudo journalctl --rotate; then
        log_verbose "Journal rotation completed successfully"
    else
        log_error "Journal rotation failed, continuing anyway..."
    fi
    
    # Clean old entries
    if sudo journalctl -u "$SERVICE" --vacuum-time="$RETAIN"; then
        log_verbose "Journal vacuum completed successfully"
    else
        log_error "Journal vacuum failed, continuing anyway..."
    fi
    
    local journal_size_after
    journal_size_after=$(get_journal_size)
    
    log_info "Journal cleaning completed"
    log_info "Size before: $journal_size_before, after: $journal_size_after"
}

# Clean ALL journal history for the service (nuclear option)
clean_all_journal() {
    log_info "WARNING: This attempts to purge journal history referencing: $SERVICE"
    log_info "NOTE: Journald vacuums are global; they cannot surgically remove only one unit's entries."
    if [[ $GLOBAL_NUKE -eq 1 ]]; then
        log_info "GLOBAL mode enabled: will additionally remove persisted files under /var/log/journal (destructive)."
    fi
    if [[ $ASSUME_YES -ne 1 && "${FORCE_CLEAN_ALL:-0}" != "1" ]]; then
        if [[ -t 0 ]]; then
            read -p "Are you sure you want to proceed? (yes/no): " -r
            if [[ $REPLY != "yes" ]]; then
                log_info "Operation cancelled"
                return 1
            fi
        else
            log_error "Refusing to run clean-all in non-interactive mode without --yes or FORCE_CLEAN_ALL=1"
            return 1
        fi
    fi

    local service_was_active=0
    if systemctl is-active --quiet "$SERVICE"; then
        service_was_active=1
        log_verbose "Stopping service temporarily..."
        sudo systemctl stop "$SERVICE"
    fi

    log_info "Rotating journal..."; sudo journalctl --rotate || log_error "journal rotate failed (continuing)"

    # First aggressive time vacuum (very small retention)
    log_info "Vacuum (time 1s) ..."; sudo journalctl --vacuum-time=1s || log_error "vacuum-time failed"

    # Secondary size vacuum to shrink residual active file
    log_info "Vacuum (size 16K) ..."; sudo journalctl --vacuum-size=16K || log_error "vacuum-size failed"

    # Optional: limit files count
    log_info "Vacuum (files 1) ..."; sudo journalctl --vacuum-files=1 || log_error "vacuum-files failed"

    # Flush buffers (so restarted service starts in a fresh file)
    if [[ $GLOBAL_NUKE -eq 1 ]]; then
        log_info "Stopping journald for global purge..."; sudo systemctl stop systemd-journald || true
        log_info "Removing /var/log/journal/*"; sudo rm -rf /var/log/journal/* || log_error "rm failed"
        log_info "Starting journald..."; sudo systemctl start systemd-journald || log_error "journald restart failed"
    else
        sudo systemctl restart systemd-journald || log_error "journald restart failed"
    fi

    if [[ $service_was_active -eq 1 ]]; then
        log_verbose "Restarting service..."
        sudo systemctl start "$SERVICE"
    fi

    log_info "Requested purge sequence completed. New logs will begin accumulating from service start time."
}

get_journal_size() {
    sudo du -sh /var/log/journal 2>/dev/null | cut -f1 || echo "unknown"
}

save_journal() {
    log_info "Saving journal for service: $SERVICE ($LINES lines → $LOG_PATH)"
    
    # Create directory if it doesn't exist
    local log_dir
    log_dir="$(dirname "$LOG_PATH")"
    if [[ ! -d "$log_dir" ]]; then
        log_verbose "Creating directory: $log_dir"
        mkdir -p "$log_dir"
    fi
    
    # Backup existing log file if it exists and is not empty
    if [[ -s "$LOG_PATH" ]]; then
        local backup_path="${LOG_PATH}.backup.$(date +%Y%m%d_%H%M%S)"
        log_verbose "Backing up existing log to: $backup_path"
        cp "$LOG_PATH" "$backup_path"
    fi
    
    # Add header to log file
    {
        echo "# Journal capture for $SERVICE"
        echo "# Generated: $(date)"
        echo "# Lines: $LINES"
        echo "# Host: $(hostname)"
        echo "#"
    } > "$LOG_PATH"
    
    # Capture journal entries
    local entry_count
    if entry_count=$(sudo journalctl -u "$SERVICE" -n "$LINES" --no-pager -o short-iso >> "$LOG_PATH" 2>/dev/null; echo "$(sudo journalctl -u "$SERVICE" -n "$LINES" --no-pager | wc -l)"); then
        log_info "Successfully saved $entry_count journal entries"
        log_verbose "Log file size: $(du -h "$LOG_PATH" | cut -f1)"
    else
        log_error "Failed to capture journal entries (service may have no logs yet)"
        echo "# No journal entries found" >> "$LOG_PATH"
    fi
}

follow_journal() {
    save_journal
    
    log_info "Following new journal entries for $SERVICE (Ctrl+C to stop)"
    echo "" | tee -a "$LOG_PATH"
    echo "# --- FOLLOWING NEW ENTRIES (started $(date)) ---" | tee -a "$LOG_PATH"
    
    # Follow new entries and append to log file
    sudo journalctl -u "$SERVICE" -f -o short-iso | tee -a "$LOG_PATH"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            clean|save|refresh|follow|clean-all)
                if [[ -n "$COMMAND" ]]; then
                    log_error "Multiple commands specified. Use only one."
                    exit 1
                fi
                COMMAND="$1"
                ;;
            -l|--lines)
                LINES="$2"
                shift
                ;;
            -r|--retain)
                RETAIN="$2"
                shift
                ;;
            -f|--file)
                LOG_PATH="$2"
                shift
                ;;
            -s|--service)
                SERVICE="$2"
                shift
                ;;
            -v|--verbose)
                VERBOSE=1
                ;;
            -y|--yes)
                ASSUME_YES=1
                ;;
            -g|--global)
                GLOBAL_NUKE=1
                ;;
            -h|--help)
                show_help
                ;;
            --lines=*)
                LINES="${1#*=}"
                ;;
            --retain=*)
                RETAIN="${1#*=}"
                ;;
            --file=*)
                LOG_PATH="${1#*=}"
                ;;
            --service=*)
                SERVICE="${1#*=}"
                ;;
            *)
                # Handle legacy positional argument (lines)
                if [[ -z "$COMMAND" && "$1" =~ ^[0-9]+$ ]]; then
                    LINES="$1"
                    COMMAND="refresh"  # Default command for backward compatibility
                else
                    log_error "Unknown argument: $1"
                    show_help
                fi
                ;;
        esac
        shift
    done
    
    # Set default command if none specified (avoid non-zero return under set -e)
    if [[ -z "$COMMAND" ]]; then
        COMMAND="refresh"
    fi
    return 0
}

# Main execution function
main() {
    parse_args "$@"

    log_verbose "(debug) Parsed command=$COMMAND assume_yes=$ASSUME_YES"
    
    echo "DEBUG: entering main with args: $*" >&2

    log_verbose "Configuration:"
    log_verbose "  Command: $COMMAND"
    log_verbose "  Service: $SERVICE"
    log_verbose "  Lines: $LINES"
    log_verbose "  Retention: $RETAIN"
    log_verbose "  Log path: $LOG_PATH"
    
    # Validations
    check_privileges
    validate_service
    validate_lines
    validate_retention
    
    # Execute command
    case "$COMMAND" in
        clean)
            clean_journal
            ;;
        clean-all)
            clean_all_journal
            ;;
        save)
            save_journal
            ;;
        refresh)
            clean_journal
            save_journal
            ;;
        follow)
            follow_journal
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            exit 1
            ;;
    esac
    
    log_info "Operation completed successfully"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi