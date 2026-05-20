#!/usr/bin/env bash

print_error() {
    echo "Error: $*" >&2
}

require_file() {
    local path="$1"
    if [[ ! -f "$path" ]]; then
        print_error "Missing file: $path"
        return 1
    fi
}

require_dir() {
    local path="$1"
    if [[ ! -d "$path" ]]; then
        print_error "Missing directory: $path"
        return 1
    fi
}

trim_whitespace() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

load_nonempty_lines() {
    local file="$1"
    local -n output_ref="$2"
    local line

    require_file "$file" || return 1

    output_ref=()
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="$(trim_whitespace "$line")"
        [[ -z "$line" || "$line" == \#* ]] && continue
        output_ref+=("$line")
    done < "$file"

    if [[ "${#output_ref[@]}" -eq 0 ]]; then
        print_error "No usable entries found in $file"
        return 1
    fi
}

load_csv_entries() {
    local file="$1"
    local -n output_ref="$2"
    local line
    local item
    local raw_items

    require_file "$file" || return 1

    output_ref=()
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="$(trim_whitespace "$line")"
        [[ -z "$line" || "$line" == \#* ]] && continue

        IFS=',' read -r -a raw_items <<< "$line"
        for item in "${raw_items[@]}"; do
            item="$(trim_whitespace "$item")"
            [[ -n "$item" ]] && output_ref+=("$item")
        done
    done < "$file"

    if [[ "${#output_ref[@]}" -eq 0 ]]; then
        print_error "No usable CSV entries found in $file"
        return 1
    fi
}

resolve_sparse_model_dir() {
    local sparse_root="$1"
    local preferred_subdir="${2:-0}"
    local candidate

    if [[ -d "$sparse_root/$preferred_subdir" ]] && [[ -n "$(find "$sparse_root/$preferred_subdir" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
        printf '%s\n' "$sparse_root/$preferred_subdir"
        return 0
    fi

    while IFS= read -r candidate; do
        if [[ -n "$(find "$candidate" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done < <(find "$sparse_root" -mindepth 1 -maxdepth 1 -type d | sort)

    if [[ -d "$sparse_root/$preferred_subdir" ]]; then
        printf '%s\n' "$sparse_root/$preferred_subdir"
        return 0
    fi

    while IFS= read -r candidate; do
        printf '%s\n' "$candidate"
        return 0
    done < <(find "$sparse_root" -mindepth 1 -maxdepth 1 -type d | sort)

    return 1
}
