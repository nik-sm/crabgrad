use crate::engine::{DiscreteLabel, FloatDataScalar, Value};
use crate::nn::cross_entropy_single;
use crate::nn::models::Classifier;
use crate::optim::Optim;
use crate::utils::try_init_logging;
use anyhow::Result;
use indicatif::ProgressBar;
use itertools::Itertools;
use rand::rng;
use rand::seq::SliceRandom;

pub struct Trainer<'a> {
    model: &'a dyn Classifier,
    optim: &'a dyn Optim,
    epochs: usize,
    batch_size: usize,
}
impl<'a> Trainer<'a> {
    pub fn new(model: &'a impl Classifier, epochs: usize, optim: &'a impl Optim, batch_size: usize) -> Self {
        Trainer { model, epochs, optim, batch_size }
    }

    pub fn fit(
        &self,
        train_data_labels: impl IntoIterator<Item = (Vec<FloatDataScalar>, DiscreteLabel)>,
        test_data_labels: Option<impl IntoIterator<Item = (Vec<FloatDataScalar>, DiscreteLabel)>>,
    ) -> Result<()> {
        // Convert data and labels to Values as needed
        fn to_values(raw: impl IntoIterator<Item = (Vec<FloatDataScalar>, DiscreteLabel)>) -> Vec<(Vec<Value>, usize)> {
            raw.into_iter().map(|(xs, y)| (xs.iter().map(Value::from).collect::<Vec<_>>(), y)).collect::<Vec<_>>()
        }
        let mut train_data_labels = to_values(train_data_labels);
        let test_data_labels = test_data_labels.map(to_values);

        if let Err(e) = try_init_logging() {
            eprintln!("Error while setting up logging: {e}")
        }

        let div = train_data_labels.len() as f64 / self.batch_size as f64;
        let mut batches_per_epoch = div as usize;
        if div.fract() > 0.0 {
            batches_per_epoch += 1;
        }
        log::info!(
            "Dataset size: {}. Epochs: {}. Batches per epoch: {}",
            train_data_labels.len(),
            self.epochs,
            batches_per_epoch
        );

        let mut rng = rng();
        for e in 0..self.epochs {
            train_data_labels.shuffle(&mut rng);
            log::info!("{:-^20}", format!("Epoch {e}"));
            let bar = ProgressBar::new(batches_per_epoch.try_into()?);
            let batches = train_data_labels.iter().chunks(self.batch_size);
            for chunk in batches.into_iter() {
                let mut loss = Value::from(0.0);
                for (data, label) in chunk {
                    let logits: Vec<Value> = self.model.forward(&data)?;
                    loss = loss + cross_entropy_single(*label, &logits);
                }

                self.optim.zero_grad();
                loss.backward();
                self.optim.step();

                bar.inc(1);
            }

            log::info!("Train acc: {}", self.model.score(&train_data_labels)?);
            if let Some(ref z) = test_data_labels {
                log::info!("Test acc: {}", self.model.score(z)?)
            }
        }

        Ok(())
    }
}
