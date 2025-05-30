use crate::impl_binary_op;
use anyhow::Result;
use anyhow::bail;
use core::f64;
use itertools::Itertools;
use rand;
use rand::Rng;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use std::cell::RefCell;
use std::collections::HashSet;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::{Add, Deref, Div, Mul, Sub};
use std::ptr;
use std::rc::Rc;

pub type FloatDataScalar = f64;
pub type IntDataScalar = i64;
pub type DiscreteLabel = usize;

pub struct Dataset<X, Y> {
    pub items: Vec<(Vec<X>, Y)>,
    pub n_features: usize,
    pub n_classes: usize,
}
impl<X: Clone, Y: Clone + Eq + Hash> Dataset<X, Y> {
    pub fn new(data: Vec<Vec<X>>, labels: Vec<Y>) -> Result<Self> {
        Self::new_from_slice(data.into_iter().zip(labels).collect::<Vec<_>>().as_ref())
    }

    fn new_from_slice(data_labels: &[(Vec<X>, Y)]) -> Result<Self> {
        let uniq_lens: Vec<usize> = data_labels.iter().map(|(x, _)| x.len()).unique().collect();
        let n_uniq_labels: usize = data_labels.iter().map(|(_, y)| y).unique().count();
        match uniq_lens.len() {
            1 => Ok(Self { items: data_labels.to_vec(), n_features: uniq_lens[0], n_classes: n_uniq_labels }),
            _ => bail!("Data with multiple lengths found: {:?}", uniq_lens),
        }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn split_at(&self, n: usize) -> Result<(Dataset<X, Y>, Dataset<X, Y>)> {
        let (left, right) = self.items.split_at(n);
        Ok((Dataset::new_from_slice(left)?, Dataset::new_from_slice(right)?))
    }

    pub fn shuffle(&mut self, rng: &mut impl Rng) {
        self.items.shuffle(rng)
    }

    #[must_use]
    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_precision_loss)]
    pub fn train_test_split(&mut self, train_frac: f64, test_frac: f64) -> Result<(Dataset<X, Y>, Dataset<X, Y>)> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let n_total = self.len() as f64;
        let n_train = ((train_frac / (train_frac + test_frac)).clamp(0.0, 1.0) * n_total) as usize;

        self.shuffle(&mut rng);
        let (train_dataset, test_dataset) = self.split_at(n_train)?;
        Ok((train_dataset, test_dataset))
    }
}

impl<X, Y> IntoIterator for Dataset<X, Y> {
    type Item = (Vec<X>, Y);

    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

#[derive(Debug, Clone)]
pub struct ValueInner {
    pub data: FloatDataScalar,
    pub grad: Option<FloatDataScalar>,
    pub backward_fn: Option<fn(&ValueInner)>,
    pub prev_nodes: Option<Vec<Value>>,
}

#[derive(Debug)]
pub struct Value(Rc<RefCell<ValueInner>>);

impl Clone for Value {
    /// To make it super clear that `value.clone()` only increments Rc
    #[inline]
    fn clone(&self) -> Self {
        Self(Rc::clone(&self.0))
    }
}

impl From<FloatDataScalar> for Value {
    #[inline]
    fn from(data: FloatDataScalar) -> Self {
        Self(Rc::new(RefCell::new(ValueInner::from(data))))
    }
}
impl From<&FloatDataScalar> for Value {
    #[inline]
    fn from(data: &FloatDataScalar) -> Self {
        Self(Rc::new(RefCell::new(ValueInner::from(data))))
    }
}

impl From<IntDataScalar> for Value {
    #[inline]
    fn from(data: IntDataScalar) -> Self {
        Self(Rc::new(RefCell::new(ValueInner::from(data))))
    }
}
impl From<&IntDataScalar> for Value {
    #[inline]
    fn from(data: &IntDataScalar) -> Self {
        Self(Rc::new(RefCell::new(ValueInner::from(data))))
    }
}

impl PartialEq for Value {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self.0.as_ptr(), other.0.as_ptr())
    }
}

impl Eq for Value {}

impl Hash for Value {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        let h: *mut ValueInner = self.0.as_ptr();
        ptr::hash(h, state);
    }
}

impl ValueInner {
    #[must_use]
    pub fn new(data: FloatDataScalar, prev_nodes: Option<Vec<Value>>, backward_fn: Option<fn(&Self)>) -> Self {
        Self { data, grad: None, prev_nodes, backward_fn }
    }
}

impl From<FloatDataScalar> for ValueInner {
    #[inline]
    fn from(data: FloatDataScalar) -> Self {
        Self { data, grad: None, backward_fn: None, prev_nodes: None }
    }
}
impl From<&FloatDataScalar> for ValueInner {
    #[inline]
    fn from(data: &FloatDataScalar) -> Self {
        Self { data: *data, grad: None, backward_fn: None, prev_nodes: None }
    }
}

impl From<IntDataScalar> for ValueInner {
    #[inline]
    fn from(data: IntDataScalar) -> Self {
        Self { data: data as FloatDataScalar, grad: None, backward_fn: None, prev_nodes: None }
    }
}
impl From<&IntDataScalar> for ValueInner {
    #[inline]
    fn from(data: &IntDataScalar) -> Self {
        Self { data: *data as FloatDataScalar, grad: None, backward_fn: None, prev_nodes: None }
    }
}

impl Deref for Value {
    type Target = Rc<RefCell<ValueInner>>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// TODO - convert from recursion to iteration
#[allow(clippy::mutable_key_type)]
fn build_topo_recursive(node: &Value, visited: &mut HashSet<Value>, topo_rev: &mut Vec<Value>) {
    // v is the child node and we have a link to our parent nodes
    if !visited.contains(node) {
        // PartialEq and Hash both use address of ValueInner
        visited.insert(node.clone());
        if let Some(prev) = &node.borrow().prev_nodes {
            for ancestor in prev {
                build_topo_recursive(ancestor, visited, topo_rev);
            }
        }
        topo_rev.push(node.clone());
    }
}

// // NOTE - iterative was much slower in a simple benchmark
// fn build_topo_iterative(root_node: Value) -> Vec<Value> {
//     enum State {
//         Unripe, // Visit this node's ancestors first
//         Ripe,   // Node's ancestors were completed. Now it is ready
//     }

//     let mut visited = HashSet::new();
//     let mut topo_rev = Vec::new(); // The actual topological order
//     let mut stack = vec![(root_node, State::Unripe)];
//     while let Some((node, state)) = stack.pop() {
//         match state {
//             State::Unripe => {
//                 stack.push((node.clone(), State::Ripe));
//                 if let Some(prev) = node.borrow().prev_nodes.clone() {
//                     for ancestor in prev {
//                         stack.push((ancestor.clone(), State::Unripe));
//                     }
//                 }
//             }
//             State::Ripe => {
//                 if !visited.contains(&node) {
//                     visited.insert(node.clone());
//                     topo_rev.push(node);
//                 }
//             }
//         }
//     }
//     topo_rev
// }

impl Value {
    #[must_use]
    pub fn new(data: FloatDataScalar, prev_nodes: Option<Vec<Self>>, backward_fn: Option<fn(&ValueInner)>) -> Self {
        Self(Rc::new(RefCell::new(ValueInner::new(data, prev_nodes, backward_fn))))
    }

    #[must_use]
    pub fn data(&self) -> FloatDataScalar {
        self.borrow().data
    }

    #[must_use]
    pub fn grad(&self) -> Option<FloatDataScalar> {
        self.borrow().grad
    }

    #[must_use]
    pub fn backward_fn(&self) -> Option<fn(&ValueInner)> {
        self.borrow().backward_fn
    }

    pub fn zero_grad(&self) {
        self.borrow_mut().grad = None;
    }

    pub fn pow<T: Into<Self>>(&self, exponent: T) -> Self {
        let exponent = exponent.into();

        let data = self.data().powf(exponent.data());
        let prev_nodes = vec![self.clone(), exponent];
        let backward_fn = |our_value_inner: &ValueInner| {
            // TODO - left off here
            match our_value_inner.prev_nodes.as_deref() {
                Some([base, exponent]) => {
                    let mut base = base.borrow_mut();
                    let mut exponent = exponent.borrow_mut();
                    let our_grad = our_value_inner.grad.unwrap_or(0.0);
                    // deriv of f = a ^ b   w.r.t.   a
                    base.grad = Some(
                        (exponent.data * base.data.powf(exponent.data - 1.0))
                            .mul_add(our_grad, base.grad.unwrap_or(0.0)),
                    );
                    // deriv of f = a ^ b   w.r.t.   b
                    exponent.grad = Some(
                        (base.data.powf(exponent.data) * base.data.ln())
                            .mul_add(our_grad, exponent.grad.unwrap_or(0.0)),
                    );
                }
                _ => {
                    unreachable!("binary op must have two ancestors")
                }
            }
        };

        Self::new(data, Some(prev_nodes), Some(backward_fn))
    }

    #[must_use]
    pub fn exp(&self) -> Self {
        Self::from(f64::consts::E).pow(self.clone())
    }

    #[must_use]
    pub fn log(&self) -> Self {
        let data = FloatDataScalar::ln(self.data());
        let prev_nodes = vec![self.clone()];
        let backward_fn = |our_value_inner: &ValueInner| match our_value_inner.prev_nodes.as_deref() {
            Some([orig]) => {
                let our_grad = our_value_inner.grad.unwrap_or(0.0);
                let mut orig = orig.borrow_mut();
                orig.grad = Some((1.0 / orig.data).mul_add(our_grad, orig.grad.unwrap_or(0.0)));
            }
            _ => {
                unreachable!("log must have one ancestor")
            }
        };
        Self::new(data, Some(prev_nodes), Some(backward_fn))
    }

    // Recursive way
    pub fn backward(&self) {
        // Topological order means for all directed edges  parent->child, parent appears first
        // To easily satisfy this property, we add each child, then add its parents, and reverse the whole list at the end

        #[allow(clippy::mutable_key_type)]
        let mut visited = HashSet::new(); // For quick lookup
        #[allow(clippy::mutable_key_type)]
        let mut topo_rev = Vec::new(); // The actual topological order
        build_topo_recursive(self, &mut visited, &mut topo_rev);

        self.borrow_mut().grad = Some(1.0);
        for v in topo_rev.iter().rev() {
            if let Some(backprop) = &v.borrow().backward_fn {
                backprop(&v.borrow());
            }
        }
    }

    // // Iterative way
    // pub fn backward(&self) {
    //     let topo = build_topo_iterative(self.clone());

    //     self.borrow_mut().grad = Some(1.0);
    //     for v in topo.iter().rev() {
    //         if let Some(backprop) = &v.borrow().backward_fn {
    //             backprop(&v.borrow());
    //         }
    //     }
    // }

    #[must_use]
    pub fn relu(&self) -> Self {
        let data = if self.data() <= 0.0 { 0.0 } else { self.data() };
        let prev_nodes = vec![self.clone()];
        let backward_fn = |our_value_inner: &ValueInner| match our_value_inner.prev_nodes.as_deref() {
            Some([first]) => {
                let mut first = first.borrow_mut();
                let our_grad: f64 = our_value_inner.grad.unwrap_or(0.0);
                let multiplier: f64 = if first.data > 0.0 { 1.0 } else { 0.0 };
                first.grad = Some(multiplier.mul_add(our_grad, first.grad.unwrap_or(0.0)));
            }
            _ => {
                unreachable!("relu must have one ancestor")
            }
        };
        Self::new(data, Some(prev_nodes), Some(backward_fn))
    }
}

#[must_use]
pub fn argmax(values: &[Value]) -> usize {
    // For now (while Value is scalar) - a separate function.
    // Even once Value contains vector - this is non-differentiable and the returned
    // Value objects do not have grad or backward_fn set

    let mut max_idx = 0;
    let mut max_val = f64::NEG_INFINITY;
    for (idx, v) in values.iter().enumerate() {
        if v.data().is_finite() && v.data() > max_val {
            max_val = v.data();
            max_idx = idx;
        }
    }
    max_idx
}

impl_binary_op!(self, rhs, Add, add, _add, +, {
    let data = self.data() + rhs.data();
    let prev_nodes = vec![self.clone(), rhs.clone()];
    let backward_fn = |our_value_inner: &ValueInner| {
        match our_value_inner.prev_nodes.as_deref() {
            Some([first, second]) => {
                let our_grad = our_value_inner.grad.unwrap_or(0.0);

                let first_grad = first.grad().unwrap_or(0.0);
                first.borrow_mut().grad = Some(first_grad + our_grad);

                let second_grad = second.grad().unwrap_or(0.0);
                second.borrow_mut().grad = Some(second_grad + our_grad);
            },
            _ => {
                unreachable!("binary op must have two ancestors")
            }
        }
    };
    Value::new(data, Some(prev_nodes), Some(backward_fn))
}
);
impl_binary_op!(self, rhs, Mul, mul, _mul, *, {
    let data = self.data() * rhs.data();
    let prev_nodes = vec![self.clone(), rhs.clone()];
    let backward_fn = |our_value_inner: &ValueInner| match our_value_inner.prev_nodes.as_deref() {
        Some([first, second]) => {
            let our_grad = our_value_inner.grad.unwrap_or(0.0);

            let first_grad = first.grad().unwrap_or(0.0);
            first.borrow_mut().grad = Some(second.data().mul_add(our_grad, first_grad));

            let second_grad = second.grad().unwrap_or(0.0);
            second.borrow_mut().grad = Some(first.data().mul_add(our_grad, second_grad));
        }
        _ => {
            unreachable!("binary op must have two ancestors")
        }
    };
    Value::new(data, Some(prev_nodes), Some(backward_fn))
});
impl_binary_op!(self, rhs, Div, div, _div, /, {
    self * rhs.pow(-1.0)
});
impl_binary_op!(self, rhs, Sub, sub, _sub, -, {
    self + (-1.0 * rhs)
});

// TODO - using these functions to implement Neuron.normalize was extremely slow - why?
#[must_use]
#[inline]
pub fn sum(values: &[Value]) -> Value {
    values.iter().fold(Value::from(0.0), |acc, val| acc + val)
}

#[must_use]
#[inline]
pub fn prod(values: &[Value]) -> Value {
    values.iter().fold(Value::from(1.0), |acc, val| acc * val)
}

#[must_use]
#[inline]
pub fn pow<T: Clone + Into<Value>>(values: &[Value], exponent: &T) -> Vec<Value> {
    values.iter().map(|value| value.pow(exponent.clone().into())).collect()
}

#[must_use]
#[inline]
pub fn exp(values: &[Value]) -> Vec<Value> {
    values.iter().map(Value::exp).collect()
}

#[must_use]
#[inline]
pub fn norm(values: &[Value]) -> Value {
    sum(&pow(values, &2)).pow(0.5)
}

#[must_use]
#[inline]
pub fn to_vec(values: &[Value]) -> Vec<FloatDataScalar> {
    values.iter().map(Value::data).collect()
}

// TODO - also impl +=, -=, etc, unary ops

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert_close, assert_vec_close};
    use anyhow::Result;
    use paste::paste;
    use tch::Tensor;

    /// Test the use of an operator on a Value, from either side
    macro_rules! test_op_data {
        ($val1:expr, $val2:expr, $name:ident, $op:tt, $result:expr) => {
            paste! {
            #[test]
                fn [<test_ $name _both>]() {
                    assert_close!((Value::from($val1) $op Value::from($val2)).borrow().data, $result)
                }
            }

            paste! {
            #[test]
                fn [<test_ $name _lhs>] () {
                    let res = ($val1 $op Value::from($val2));
                    let b = res.borrow();
                    assert_close!(b.data, $result)
                }
            }

            paste! {
                #[test]
                fn [<test_ $name _rhs>] () {
                    assert_close!((Value::from($val1) $op $val2).borrow().data, $result)
                }
            }
        };
    }

    test_op_data!(5.0, 6.0, add, +, 11.0);
    test_op_data!(5.0, 6.0, sub, -, -1.0);
    test_op_data!(5.0, 6.0, mul, *, 30.0);
    test_op_data!(5.0, 6.0, div, /, 5./6.);

    #[test]
    fn check_torch_install() -> Result<()> {
        let t = Tensor::from_slice(&[3, 1, 4, 1, 5]);
        let t = t * 2;
        assert_eq!(Vec::<i64>::try_from(t)?, vec![6, 2, 8, 2, 10]);
        Ok(())
    }

    #[test]
    fn compare_torch_1() {
        let x = Value::from(-2.0);
        let y = x.clone() * x.clone();
        y.backward();
        let (xmg, ymg) = (x, y);

        let x = Tensor::from(-2.0).set_requires_grad(true);
        let y = &x * &x;
        y.backward();
        let (xpt, ypt) = (&x, y);

        //  forward pass went well
        assert_eq!(ymg.data(), ypt.double_value(&[]));
        // // backward pass went well
        assert_eq!(xmg.grad().unwrap(), xpt.grad().double_value(&[]));
    }

    #[test]
    fn compare_torch_2() {
        let x = Value::from(-2.0);
        let y = x.clone() + x.clone();
        y.backward();
        let (xmg, ymg) = (x, y);

        let x = Tensor::from(-2.0).set_requires_grad(true);
        let y = &x + &x;
        y.backward();
        let (xpt, ypt) = (&x, y);

        //  forward pass went well
        assert_eq!(ymg.data(), ypt.double_value(&[]));
        // // backward pass went well
        assert_eq!(xmg.grad().unwrap(), xpt.grad().double_value(&[]));
    }

    #[test]
    fn compare_torch_many1() {
        let x = Value::from(-4.0);
        let z = 2 * &x + 2 + &x;
        let q = &z.relu() + &z * &x;
        let h = (&z * &z).relu();
        let y = h + &q + &q * &x;
        y.backward();
        let (xmg, ymg) = (&x, y);

        let x = Tensor::from(-4.0).set_requires_grad(true);
        let z: Tensor = 2.0 * &x + 2.0 + &x;
        let q = &z.relu() + &z * &x;
        let h = (&z * &z).relu();
        let y = h + &q + &q * &x;
        y.backward();
        let (xpt, ypt) = (&x, y);

        //  forward pass went well

        assert_eq!(ymg.data(), ypt.double_value(&[]));
        // // backward pass went well
        assert_eq!(xmg.grad().unwrap(), xpt.grad().double_value(&[]));
    }

    #[test]
    fn compare_torch_many2() {
        let a = Value::from(-4.0);
        let b = Value::from(2.0);
        let c = &a + &b;
        let d = &a * &b + &b.pow(3.0);
        let c = &c + &c + 1.0;
        let c = &c + 1 + c + (-1 * &a);
        let d = &d + &d * 2 + (&b + &a).relu();
        let d = &d + 3 * &d + (&b - &a).relu();
        let e = &c - &d;
        let f = e.pow(2.0);
        let g = &f / Value::from(2.0);
        let g = &g + 10.0 / f;
        let h = g.log();
        h.backward();
        let (amg, bmg, gmg) = (a, b, g);

        let a = Tensor::from(-4.0).set_requires_grad(true);
        let b = Tensor::from(2.0).set_requires_grad(true);
        let c = &a + &b;
        let d = &a * &b + &b.pow(&Tensor::from(3.0));
        let c = &c + &c + 1;
        let c = &c + 1 + &c + (-&a);
        let d = &d + &d * 2 + (&b + &a).relu();
        let d = &d + 3 * &d + (&b - &a).relu();
        let e: Tensor = &c - &d;
        let f = e.pow(&Tensor::from(2.0));
        let g = &f / &Tensor::from(2.0);
        let g: Tensor = &g + &Tensor::from(10.0) / f;
        let h = g.log();
        h.backward();
        let (apt, bpt, gpt) = (a, b, g);

        // forward pass went well
        assert_close!(gmg.data(), gpt.double_value(&[]));

        // backward pass went well
        // TODO - note that even strict equality is working, indicating probably op-for-op equivalence
        // When we would be satisfied with merely achieving assert_close
        assert_close!(amg.grad().unwrap(), apt.grad().double_value(&[]));
        assert_close!(bmg.grad().unwrap(), bpt.grad().double_value(&[]));
    }

    #[test]
    fn verify_hashset_behavior() {
        use std::collections::HashSet;

        let mut visited = HashSet::new();

        let node = Value::from(5.0);
        let mimic = Value::from(5.0);

        visited.insert(&node);

        assert!(visited.contains(&node));
        // Demonstrate that we are not comparing by value
        assert!(!visited.contains(&mimic));

        // Demonstrate that we are comparing by reference (address)
        let clone = node.clone();
        assert!(node == clone);
        assert!(visited.contains(&clone));
    }

    #[test]
    fn test_sum() {
        let values = vec![Value::from(2.0), Value::from(3.0), Value::from(-1.0)];
        assert_close!(sum(&values).data(), 4.0);
    }

    #[test]
    fn test_prod() {
        let values = vec![Value::from(2.0), Value::from(3.0), Value::from(-1.0)];
        assert_close!(prod(&values).data(), -6.0);
    }

    #[test]
    fn test_pow() {
        let values = vec![Value::from(2.0), Value::from(3.0), Value::from(-1.0)];
        assert_vec_close!(pow(&values, &2.0), vec![Value::from(4.0), Value::from(9.0), Value::from(1.0)]);
    }

    #[test]
    fn test_exp() {
        let values = vec![Value::from(0.5f64.ln()), Value::from(0.5f64.ln())];
        assert_vec_close!(exp(&values), vec![Value::from(0.5), Value::from(0.5)]);
    }

    #[test]
    fn test_to_vec() {
        let values = vec![Value::from(1.0), Value::from(2.0)];
        assert_eq!(to_vec(&values), vec![1.0, 2.0]);
        dbg!(values);
    }

    #[test]
    fn test_norm() {
        let values = vec![Value::from(2.0), Value::from(3.0), Value::from(-1.0)];
        assert_close!(
            norm(&values).data(),
            values.iter().map(|v| v.pow(2.0)).fold(0.0, |acc, val| acc + val.data()).sqrt()
        );
    }

    // #[test]
    // fn compare_topos() {
    //     fn print_nodes(nodes: impl IntoIterator<Item = Value>) {
    //         nodes.into_iter().for_each(|node| print!("{}, ", node.data()));
    //     }

    //     let x = Value::from(-4.0);
    //     let z = 2 * &x + 2 + &x;
    //     let q = &z.relu() + &z * &x;
    //     let h = (&z * &z).relu();
    //     let y = h + &q + &q * &x;
    //     y.backward();

    //     let topo_iter = build_topo_iterative(y.clone());

    //     let x = Value::from(-4.0);
    //     let z = 2 * &x + 2 + &x;
    //     let q = &z.relu() + &z * &x;
    //     let h = (&z * &z).relu();
    //     let y = h + &q + &q * &x;
    //     y.backward();

    //     let mut visited = HashSet::new();
    //     let mut topo_rev_rec = Vec::new();
    //     build_topo_recursive(&y.clone(), &mut visited, &mut topo_rev_rec);

    //     print!("Iter: ");
    //     print_nodes(topo_iter.into_iter().rev());
    //     println!();
    //     print!("Rec: ");
    //     print_nodes(topo_rev_rec.into_iter().rev());
    // }
}
