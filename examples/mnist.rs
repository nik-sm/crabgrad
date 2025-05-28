use anyhow::Result;
use crabgrad::{
    engine::{Dataset, DiscreteLabel, FloatDataScalar},
    nn::{Module, Trainer, MLP},
    optim::AdamW,
};
use hf_hub::{api::sync::Api, Repo, RepoType};
use parquet::file::reader::SerializedFileReader;
// use parquet::file::reader::FileReader;

fn load_parquet(
    parquet: SerializedFileReader<std::fs::File>,
    take_n: usize,
) -> Result<(Vec<Vec<FloatDataScalar>>, Vec<DiscreteLabel>)> {
    // let samples = parquet.metadata().file_metadata().num_rows() as usize;
    let mut buffer_images: Vec<Vec<FloatDataScalar>> = Vec::with_capacity(take_n);
    let mut buffer_labels: Vec<DiscreteLabel> = Vec::with_capacity(take_n);

    for row in parquet.into_iter().take(take_n).flatten() {
        for (_name, field) in row.get_column_iter() {
            if let parquet::record::Field::Group(subrow) = field {
                for (_name, field) in subrow.get_column_iter() {
                    if let parquet::record::Field::Bytes(value) = field {
                        let image: Vec<FloatDataScalar> = image::load_from_memory(value.data())
                            .unwrap()
                            .to_luma8()
                            .into_raw()
                            .iter()
                            .map(|&pixel| (pixel as FloatDataScalar / 255.0))
                            .collect();
                        buffer_images.push(image);
                    }
                }
            } else if let parquet::record::Field::Long(label) = field {
                buffer_labels.push(*label as DiscreteLabel);
            }
        }
    }

    dbg!("buffer lens", buffer_images.len(), buffer_labels.len());

    Ok((buffer_images, buffer_labels))
}

fn load_mnist(
    n_train: usize,
    n_test: usize,
) -> Result<(Dataset<FloatDataScalar, DiscreteLabel>, Dataset<FloatDataScalar, DiscreteLabel>)> {
    let api = Api::new()?;
    let dataset_id = "ylecun/mnist".to_string();
    let repo = Repo::with_revision(dataset_id, RepoType::Dataset, "refs/convert/parquet".to_string());
    let repo = api.repo(repo);

    let test_parquet_filename = repo.get("mnist/test/0000.parquet")?;
    let train_parquet_filename = repo.get("mnist/train/0000.parquet")?;
    let test_parquet = SerializedFileReader::new(std::fs::File::open(test_parquet_filename)?)?;
    let train_parquet = SerializedFileReader::new(std::fs::File::open(train_parquet_filename)?)?;

    let (train_images, train_labels) = load_parquet(train_parquet, n_train)?;
    let (test_images, test_labels) = load_parquet(test_parquet, n_test)?;

    Ok((Dataset::new(train_images, train_labels)?, Dataset::new(test_images, test_labels)?))
}

fn main() -> Result<()> {
    let (train_data_labels, test_data_labels) = load_mnist(5_000, 5_000)?;

    let epochs = 2;
    let batch_size = 32;
    let model = MLP::new(28 * 28, &[], 10, true);
    let mut optim = AdamW::new(model.parameters(), 1e-3, 0.9, 0.999, 1e-8, 0.0);
    let mut trainer = Trainer::new(&model, &mut optim, epochs, batch_size);
    trainer.fit(train_data_labels, Some(test_data_labels))?;
    Ok(())
}
