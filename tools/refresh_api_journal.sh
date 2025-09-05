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
#   reset      - Stop service, clean journals, restart service (most effective cleanup)
#   status     - Show service status and recent journal entries (diagnostic)
#   nuke       - EXTREME: Stop service, clean all journals, disable service to prevent new logs
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
#   ./refresh_api_journal.sh reset --yes
#   ./refresh_api_journal.sh status
#   ./refresh_api_journal.sh nuke --yes

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
    log_info "NOTE: Journald vacuums are global and cannot surgically remove only one unit's entries."
    log_info "       This operation will stop the service, clean global journals, and restart the service."
    
    if [[ $GLOBAL_NUKE -eq 1 ]]; then
        log_info "GLOBAL mode enabled: will remove ALL persisted journal files (EXTREMELY destructive)."
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

    # Stop the service first to prevent new log entries during cleanup
    local service_was_active=0
    if systemctl is-active --quiet "$SERVICE"; then
        service_was_active=1
        log_info "Stopping service temporarily to prevent new log entries..."
        sudo systemctl stop "$SERVICE"
        # Wait a moment for the service to fully stop
        sleep 2
    else
        log_info "Service is already stopped - good for cleaning"
    fi

    # Get initial journal usage info
    local journal_usage_before
    journal_usage_before=$(sudo journalctl --disk-usage 2>/dev/null | grep -oE '[0-9.]+[KMGT]?B' | tail -1 || echo "unknown")
    log_info "Journal disk usage before cleanup: $journal_usage_before"

    # Force journal rotation to move current entries to archived files
    log_info "Rotating journal files..."
    sudo journalctl --rotate || log_error "Journal rotation failed (continuing)"
    sleep 1

    # Aggressive cleanup strategies
    log_info "Vacuum (time-based: keeping only 1 second)..."
    sudo journalctl --vacuum-time=1s || log_error "vacuum-time failed"

    log_info "Vacuum (size-based: keeping only 16KB)..."
    sudo journalctl --vacuum-size=16K || log_error "vacuum-size failed"

    log_info "Vacuum (file-based: keeping only 1 file)..."
    sudo journalctl --vacuum-files=1 || log_error "vacuum-files failed"

    # Global nuclear option if requested
    if [[ $GLOBAL_NUKE -eq 1 ]]; then
        log_info "GLOBAL NUKE: Stopping systemd-journald..."
        sudo systemctl stop systemd-journald || true
        sleep 1
        
        log_info "GLOBAL NUKE: Removing all persistent journal files..."
        sudo rm -rf /var/log/journal/* || log_error "Failed to remove journal files"
        sudo rm -rf /run/log/journal/* || true
        
        log_info "GLOBAL NUKE: Restarting systemd-journald..."
        sudo systemctl start systemd-journald || log_error "Failed to restart journald"
        sleep 2
    else
        # Standard approach: restart journald to flush buffers
        log_info "Restarting systemd-journald to flush buffers..."
        sudo systemctl restart systemd-journald || log_error "journald restart failed"
        sleep 2
    fi

    # Restart the service if it was running
    if [[ $service_was_active -eq 1 ]]; then
        log_info "Restarting service..."
        sudo systemctl start "$SERVICE"
        sleep 2
        
        # Verify service started successfully
        if systemctl is-active --quiet "$SERVICE"; then
            log_info "Service restarted successfully"
        else
            log_error "Service failed to restart properly"
            sudo systemctl status "$SERVICE" --no-pager -l || true
        fi
    fi

    # Show final journal usage
    local journal_usage_after
    journal_usage_after=$(sudo journalctl --disk-usage 2>/dev/null | grep -oE '[0-9.]+[KMGT]?B' | tail -1 || echo "unknown")
    log_info "Journal disk usage after cleanup: $journal_usage_after"

    # Check if we actually cleaned the service logs
    local remaining_entries
    remaining_entries=$(sudo journalctl -u "$SERVICE" --no-pager -q | wc -l || echo "0")
    
    if [[ $remaining_entries -eq 0 ]]; then
        log_info "✅ SUCCESS: No journal entries remain for $SERVICE"
    else
        log_info "⚠️  NOTE: $remaining_entries journal entries still exist for $SERVICE"
        log_info "   This is normal if the service restarted and is generating new logs."
        log_info "   To verify cleanup, check that timestamps start from service restart time."
    fi

    log_info "Journal purge sequence completed."
}

# Reset service and clean journals (most effective cleanup)
reset_service_journal() {
    log_info "RESET: This will stop $SERVICE, clean journals, and restart the service"
    log_info "       This is the most effective way to clear service-specific logs."
    
    if [[ $ASSUME_YES -ne 1 ]]; then
        if [[ -t 0 ]]; then
            read -p "Proceed with service reset? (yes/no): " -r
            if [[ $REPLY != "yes" ]]; then
                log_info "Operation cancelled"
                return 1
            fi
        else
            log_error "Reset requires confirmation. Use --yes for non-interactive mode."
            return 1
        fi
    fi

    # Check current service status
    local service_status
    service_status=$(systemctl is-active "$SERVICE" || echo "inactive")
    log_info "Current service status: $service_status"

    # Stop the service
    log_info "Stopping $SERVICE..."
    sudo systemctl stop "$SERVICE" || log_error "Failed to stop service"
    
    # Wait for complete shutdown
    sleep 3
    
    # Verify service is stopped
    if systemctl is-active --quiet "$SERVICE"; then
        log_error "Service is still running after stop command"
        return 1
    fi
    
    log_info "✅ Service stopped successfully"

    # Clean journals aggressively
    log_info "Cleaning journal files..."
    
    # Rotate to create archives
    sudo journalctl --rotate || log_error "Rotation failed"
    
    # Multiple cleanup passes
    sudo journalctl --vacuum-time=1s || log_error "Time vacuum failed"
    sudo journalctl --vacuum-size=1K || log_error "Size vacuum failed" 
    sudo journalctl --vacuum-files=1 || log_error "File vacuum failed"
    
    # Restart journald to ensure clean state
    log_info "Restarting systemd-journald..."
    sudo systemctl restart systemd-journald || log_error "Failed to restart journald"
    sleep 2

    # Restart the service
    log_info "Starting $SERVICE..."
    sudo systemctl start "$SERVICE" || {
        log_error "Failed to start service"
        sudo systemctl status "$SERVICE" --no-pager -l || true
        return 1
    }
    
    # Wait for service to initialize
    sleep 3
    
    # Verify service is running
    if systemctl is-active --quiet "$SERVICE"; then
        log_info "✅ Service restarted successfully"
    else
        log_error "❌ Service failed to start properly"
        sudo systemctl status "$SERVICE" --no-pager -l || true
        return 1
    fi

    # Show clean state
    local current_entries
    current_entries=$(sudo journalctl -u "$SERVICE" --no-pager -q | wc -l || echo "0")
    log_info "Journal entries for $SERVICE: $current_entries (should be minimal/recent)"
    
    if [[ $current_entries -gt 0 ]]; then
        log_info "Recent log entries (showing first 5):"
        sudo journalctl -u "$SERVICE" -n 5 --no-pager || true
    fi
    
    log_info "✅ Service reset completed successfully"
}

# Nuclear option: Stop service, clean all journals, keep service disabled
nuke_service_journal() {
    log_info "🚨 NUCLEAR OPTION: This will completely stop $SERVICE and disable it to prevent log floods"
    log_info "       Use this when a broken service is generating logs faster than they can be cleaned."
    log_info "       The service will remain DISABLED after cleanup - you'll need to manually fix and re-enable it."
    
    if [[ $ASSUME_YES -ne 1 ]]; then
        if [[ -t 0 ]]; then
            read -p "This will DISABLE the service after cleanup. Continue? (yes/no): " -r
            if [[ $REPLY != "yes" ]]; then
                log_info "Operation cancelled"
                return 1
            fi
        else
            log_error "Nuke requires confirmation. Use --yes for non-interactive mode."
            return 1
        fi
    fi

    # Check current service status
    local service_status
    service_status=$(systemctl is-active "$SERVICE" || echo "inactive")
    local service_enabled
    service_enabled=$(systemctl is-enabled "$SERVICE" || echo "unknown")
    
    log_info "Current service status: $service_status"
    log_info "Current service enabled: $service_enabled"

    # Stop and disable the service to prevent restart floods
    log_info "🛑 Stopping and disabling $SERVICE to prevent log floods..."
    sudo systemctl stop "$SERVICE" || log_error "Failed to stop service (continuing)"
    sudo systemctl disable "$SERVICE" || log_error "Failed to disable service (continuing)"
    
    # Wait for complete shutdown and ensure no restart
    sleep 5
    
    # Verify service is fully stopped
    local current_status
    current_status=$(systemctl is-active "$SERVICE" || echo "inactive")
    if [[ "$current_status" != "inactive" ]]; then
        log_error "Service is still $current_status - forcing kill"
        sudo systemctl kill "$SERVICE" || log_error "Failed to kill service"
        sleep 2
    fi
    
    log_info "✅ Service stopped and disabled"

    # Get initial journal statistics
    local total_entries_before
    total_entries_before=$(sudo journalctl -u "$SERVICE" --no-pager -q | wc -l || echo "0")
    
    local journal_usage_before
    journal_usage_before=$(sudo journalctl --disk-usage 2>/dev/null | grep -oE '[0-9.]+[KMGT]?B' | tail -1 || echo "unknown")
    
    log_info "Journal entries before cleanup: $total_entries_before"
    log_info "Total journal disk usage: $journal_usage_before"

    # Aggressive multi-pass cleanup
    log_info "🧹 Starting aggressive journal cleanup (multiple passes)..."
    
    # Pass 1: Rotate and vacuum by time
    log_info "Pass 1: Rotating journals and time-based vacuum..."
    sudo journalctl --rotate || log_error "Rotation failed"
    sleep 1
    sudo journalctl --vacuum-time=1s || log_error "Time vacuum failed"
    
    # Pass 2: Size-based vacuum (very aggressive)
    log_info "Pass 2: Size-based vacuum..."
    sudo journalctl --vacuum-size=1K || log_error "Size vacuum failed"
    
    # Pass 3: File count vacuum
    log_info "Pass 3: File count vacuum..."
    sudo journalctl --vacuum-files=1 || log_error "File vacuum failed"
    
    # Pass 4: Restart journald to flush any remaining buffers
    log_info "Pass 4: Restarting journald to flush buffers..."
    sudo systemctl restart systemd-journald || log_error "Failed to restart journald"
    sleep 3
    
    # Pass 5: Final cleanup pass
    log_info "Pass 5: Final cleanup pass..."
    sudo journalctl --rotate || true
    sudo journalctl --vacuum-time=1s || true
    sudo journalctl --vacuum-size=1K || true
    
    # Check final results
    local total_entries_after
    total_entries_after=$(sudo journalctl -u "$SERVICE" --no-pager -q | wc -l || echo "0")
    
    local journal_usage_after
    journal_usage_after=$(sudo journalctl --disk-usage 2>/dev/null | grep -oE '[0-9.]+[KMGT]?B' | tail -1 || echo "unknown")
    
    log_info "📊 CLEANUP RESULTS:"
    log_info "  Entries before: $total_entries_before"
    log_info "  Entries after:  $total_entries_after"
    log_info "  Reduction:      $((total_entries_before - total_entries_after)) entries removed"
    log_info "  Disk usage before: $journal_usage_before"
    log_info "  Disk usage after:  $journal_usage_after"
    
    if [[ $total_entries_after -eq 0 ]]; then
        log_info "🎉 SUCCESS: All journal entries for $SERVICE have been completely removed!"
    elif [[ $total_entries_after -lt 10 ]]; then
        log_info "✅ SUCCESS: Journal entries reduced to minimal level ($total_entries_after remaining)"
        log_info "   Remaining entries (showing all):"
        sudo journalctl -u "$SERVICE" --no-pager || true
    else
        log_info "⚠️  PARTIAL SUCCESS: Reduced entries from $total_entries_before to $total_entries_after"
        log_info "   Recent entries (last 5):"
        sudo journalctl -u "$SERVICE" -n 5 --no-pager || true
    fi
    
    echo ""
    log_info "🔧 POST-CLEANUP ACTIONS NEEDED:"
    log_info "   1. Fix the underlying service issue (ModuleNotFoundError, etc.)"
    log_info "   2. Re-enable the service: sudo systemctl enable $SERVICE"
    log_info "   3. Start the service: sudo systemctl start $SERVICE"
    log_info "   4. Monitor for proper startup: $0 status"
    
    log_info "🎯 Service is now DISABLED and journals are cleaned. No new logs will accumulate until you fix and re-enable the service."
}

# Show service status and diagnostics
show_service_status() {
    log_info "Service status and diagnostic information for: $SERVICE"
    echo ""
    
    # Service status
    local service_status
    service_status=$(systemctl is-active "$SERVICE" 2>/dev/null || echo "unknown")
    local service_enabled
    service_enabled=$(systemctl is-enabled "$SERVICE" 2>/dev/null || echo "unknown")
    
    echo "🔧 SERVICE STATUS:"
    echo "  Status: $service_status"
    echo "  Enabled: $service_enabled"
    echo ""
    
    # Detailed status
    echo "📋 DETAILED STATUS:"
    sudo systemctl status "$SERVICE" --no-pager -l || true
    echo ""
    
    # Journal statistics
    local total_entries
    total_entries=$(sudo journalctl -u "$SERVICE" --no-pager -q | wc -l 2>/dev/null || echo "0")
    
    local journal_size
    journal_size=$(sudo journalctl -u "$SERVICE" --no-pager -q | wc -c 2>/dev/null | numfmt --to=iec 2>/dev/null || echo "unknown")
    
    echo "📊 JOURNAL STATISTICS:"
    echo "  Total entries: $total_entries"
    echo "  Approximate size: $journal_size"
    echo ""
    
    # Recent entries
    if [[ $total_entries -gt 0 ]]; then
        echo "📝 RECENT LOG ENTRIES (last 10):"
        sudo journalctl -u "$SERVICE" -n 10 --no-pager || true
        echo ""
        
        # First entry (to show when logging started)
        echo "🕐 FIRST LOG ENTRY:"
        sudo journalctl -u "$SERVICE" --no-pager | head -1 || true
        echo ""
    else
        echo "📝 No journal entries found for this service."
        echo ""
    fi
    
    # Service issues detection
    if [[ "$service_status" != "active" ]]; then
        echo "⚠️  SERVICE ISSUES DETECTED:"
        echo "  Service is not running ($service_status)"
        
        # Check for common error patterns
        local error_patterns=("ModuleNotFoundError" "ImportError" "Permission denied" "Address already in use" "Failed to bind")
        for pattern in "${error_patterns[@]}"; do
            if sudo journalctl -u "$SERVICE" -n 50 --no-pager | grep -qi "$pattern"; then
                echo "  - Found '$pattern' in recent logs"
            fi
        done
        
        echo ""
        echo "💡 SUGGESTIONS:"
        echo "  - Check service configuration and dependencies"
        echo "  - Review recent log entries above for error details"
        echo "  - Try: systemctl restart $SERVICE"
        echo "  - For Python services, verify virtual environment and module installation"
        echo ""
    fi
    
    # Cleanup suggestions
    if [[ $total_entries -gt 1000 ]]; then
        echo "🧹 CLEANUP SUGGESTIONS:"
        echo "  Journal has $total_entries entries - consider cleaning:"
        echo "  - Light cleanup: $0 clean"
        echo "  - Full reset: $0 reset --yes"
        echo ""
    fi
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
            clean|save|refresh|follow|clean-all|reset|status|nuke)
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
        reset)
            reset_service_journal
            ;;
        nuke)
            nuke_service_journal
            ;;
        status)
            show_service_status
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