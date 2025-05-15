use anyhow::Result;
use micrograd_rs::nn::{MLP, Trainer};
use micrograd_rs::optim::SGD;
use micrograd_rs::utils::{make_binary_classification, train_test_split};

fn main() -> Result<()> {
    let n_features = 64;
    let n_classes = 2;
    let epochs = 10;
    let batch_size = 32;

    let (data, labels) = make_binary_classification(1000, n_features);

    let (train_data_labels, test_data_labels) =
        train_test_split(data.into_iter().zip(labels).collect::<Vec<_>>(), 0.8, 0.2);

    let model = MLP::new(n_features, vec![32, 16, 8], n_classes, true);
    let optim = SGD::new(&model, 0.1);
    let trainer = Trainer::new(&model, epochs, &optim, batch_size);
    trainer.fit(train_data_labels, Some(test_data_labels))?;

    Ok(())
}
