use anyhow::Result;
use criterion::{Criterion, criterion_group, criterion_main};
use micrograd_rs::engine::{Value, sum};
use micrograd_rs::nn::{Layer, Module};
use micrograd_rs::optim::{Optim, SGD};

fn ops_in_loop(n: usize) {
    let mut value = Value::from(1.0);
    for _ in 0..n {
        value = value + 1.0;
    }
}

fn layer_sgd(n: usize) -> Result<()> {
    {
        // Check that each neuron in a single layer will move as expected
        // Same strategy as used for single neuron case
        let mut layer = Layer::new(3, 2, false, false);
        layer.normalize();

        let x = vec![Value::from(1.0), Value::from(0.0), Value::from(0.0)];
        let mut optim = SGD::new(layer.parameters(), 0.1);

        for _ in 0..n {
            let out = layer.forward(&x)?;
            let loss = 1 - sum(&out);

            optim.zero_grad();
            loss.backward();
            optim.step();

            // Normalize to unit length
            layer.normalize();
        }
    }
    Ok(())
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("basic-benchmarks");
    group.sample_size(100);
    group.bench_function("ops 10_000", |b| b.iter(|| ops_in_loop(10_000)));
    group.bench_function("layer_sgd 10_000", |b| b.iter(|| layer_sgd(10_000)));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
