use anyhow::Result;
use criterion::{Criterion, criterion_group, criterion_main};
use micrograd_rs::engine::{Value, sum};
use micrograd_rs::nn::{MLP, Module};
use micrograd_rs::optim::{Optim, SGD};

fn ops_in_loop(n: usize) {
    let mut value = Value::from(1.0);
    for _ in 0..n {
        value = value + 1.0;
    }
}

fn mlp_sgd(n: usize) -> Result<()> {
    {
        let mut model = MLP::new(10, &[10], 2, true);
        let data: Vec<Value> = (0..10).map(|_| Value::from(12.345)).collect();
        let mut optim = SGD::new(model.parameters(), 0.1);

        for _ in 0..n {
            let out = model.forward(&data)?;
            let loss = 1 - sum(&out);

            optim.zero_grad();
            loss.backward();
            optim.step();

            // Normalize to unit length
            model.normalize();
        }
    }
    Ok(())
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("basic-benchmarks");
    group.sample_size(100);
    group.bench_function("ops 10_000", |b| b.iter(|| ops_in_loop(10_000)));
    group.bench_function("mlp_sgd 10_000", |b| b.iter(|| mlp_sgd(10_000)));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
