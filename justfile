set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

build:
    cargo build --release --features mlx

verify:
    cargo fmt --all --check
    cargo test --all-targets
    cargo clippy --all-targets -- -D warnings

check-native:
    cargo check --features mlx

clean:
    cargo clean
