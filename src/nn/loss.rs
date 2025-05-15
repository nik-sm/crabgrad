use crate::engine::Value;

pub fn cross_entropy(labels: &[Value], logits: &[Value]) -> Value {
    dbg!(&labels, &logits);
    unimplemented!()
}
