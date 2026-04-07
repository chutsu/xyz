#!/bin/bash
set -e

run_command() {
  TARGET="dev"
  tmux send-keys -t $TARGET -R C-l C-m
  tmux send-keys -t $TARGET -R "clear && $1" C-m C-m
}

# CMD="cargo build"
# CMD="cargo test"
CMD="cd ~/code/xyz/rust && cargo fmt && cargo test -- --nocapture"
run_command "$CMD"
