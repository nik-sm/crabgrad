use crate::engine::Value;

pub fn cross_entropy(labels: &Vec<Value>, logits: &Vec<Value>) -> Value {
    dbg!(&labels, &logits);
    unimplemented!()
}

pub fn cross_entropy_single(label: &Value, logits: &Vec<Value>) -> Value {
    dbg!(&label, &logits);
    unimplemented!()
}

pub fn softmax(logits: Vec<Value>) -> Vec<Value> {
    dbg!(&logits);
    todo!()
}
