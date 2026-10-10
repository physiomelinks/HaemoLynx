# Sourced by pre-commit and scripts/stamp-tests.sh. Not a hook itself.
# The stamp records that the tests passed for one exact staged tree, in one
# interpreter and dependency set. It lives under .git/, so it is not tracked.
# It is only valid while tracked files have no unstaged edits.

stamp_file() { git rev-parse --git-path hook-stamp/fast-tests-ok; }

# Prints the key, or nothing (status 1) when the tree has unstaged edits to
# tracked files, since the tests would then see something the index does not.
stamp_key() {
    git diff --quiet || return 1
    printf 'tree=%s\npy=%s\ndeps=%s\n' \
        "$(git write-tree)" \
        "$(realpath -s "$(command -v "$PY")")" \
        "$("$PY" -m pip freeze 2>/dev/null | sha256sum | cut -d' ' -f1)"
}

stamp_matches() {
    local f key
    f=$(stamp_file); [ -f "$f" ] || return 1
    key=$(stamp_key) || return 1
    [ "$key" = "$(cat "$f")" ]
}

stamp_write() {
    local f key
    key=$(stamp_key) || return 0
    f=$(stamp_file); mkdir -p "$(dirname "$f")"
    printf '%s\n' "$key" > "$f"
}
