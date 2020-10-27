# Pulse-Doppler radar signal processing

A very simple example of a MPD signal processing chain:

1. Generation of sampled video
2. Pulse compression
3. Doppler filtering (FFT) into the range-frequency domain
4. Constant false alarm rate processing
5. Range and velocity resolving

## Enjoy

```
cargo run --example main --release -- --files

python ./examples/plot.py
```