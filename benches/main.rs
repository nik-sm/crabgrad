use anyhow::Result;
use criterion::{Criterion, criterion_group, criterion_main};
use micrograd_rs::engine::{Value, norm, sum};
use micrograd_rs::nn::{MLP, Module as _};
use micrograd_rs::optim::{Optim as _, SGD};

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
        let mut optim = SGD::new(&model.parameters(), 0.1);

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

fn norm_a(n: usize) {
    let mut data = vec![Value::from(1.0), Value::from(2.0)];
    for _ in 0..n {
        let norm = data.iter().fold(0.0, |acc, val| val.data().mul_add(val.data(), acc)).sqrt();
        data = data.iter().map(|d| d / norm.clone()).collect();
    }
}

fn norm_b(n: usize) {
    let mut data = vec![Value::from(1.0), Value::from(2.0)];
    for _ in 0..n {
        let norm = norm(&data);
        data = data.iter().map(|d| d / norm.clone()).collect();
    }
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("basic-benchmarks");
    group.sample_size(100);
    group.bench_function("ops 10_000", |b| b.iter(|| ops_in_loop(10_000)));
    group.bench_function("mlp_sgd 10_000", |b| b.iter(|| mlp_sgd(10_000)));
    group.bench_function("norm_a 100", |b| b.iter(|| norm_a(100)));
    group.bench_function("norm_b 100", |b| b.iter(|| norm_b(100)));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
