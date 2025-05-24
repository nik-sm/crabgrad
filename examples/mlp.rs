use anyhow::Result;
use micrograd_rs::nn::{MLP, Module, Trainer};
// use micrograd_rs::optim::SGD;
use micrograd_rs::optim::AdamW;
use micrograd_rs::utils::{init_logging, make_binary_classification, train_test_split};

fn main() -> Result<()> {
    let n_features = 64;
    let n_classes = 2;
    let n_samples_each_class = 1000;
    let epochs = 1;
    let batch_size = 32;

    let (data, labels) = make_binary_classification(n_samples_each_class, n_features);

    init_logging();
    let (train_data_labels, test_data_labels) =
        train_test_split(data.into_iter().zip(labels).collect::<Vec<_>>(), 0.8, 0.2);

    // NOTE - performance for this toy problem is fragile and sensitive to hidden dims and weight init
    let model = MLP::new(n_features, vec![32], n_classes, true);
    log::info!("Number parameters: {}", model.parameters().len());
    // let optim = SGD::new(model.parameters(), 1e-3);
    let mut optim = AdamW::new(model.parameters(), 1e-2, 0.9, 0.999, 1e-8, 0.0);
    let mut trainer = Trainer::new(&model, &mut optim, epochs, batch_size);
    trainer.fit(train_data_labels, Some(test_data_labels))?;

    Ok(())
}
