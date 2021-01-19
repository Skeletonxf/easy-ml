#!/usr/bin/env bash
cargo build --target wasm32-unknown-unknown && cargo build --target wasm32-unknown-unknown --features serde
