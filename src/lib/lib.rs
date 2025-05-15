pub mod engine;
pub use engine::{Value, argmax};

pub mod nn;

pub mod ops;
pub mod utils;

pub mod optim;
pub use optim::{Optim, SGD};

#[cfg(test)]
mod tests {
    use crate::engine::Value;
    use crate::nn::{MLP, Trainer};
    use crate::utils::make_binary_classification;
    use crate::{SGD, assert_close, assert_not_close};
    use anyhow::Result;
    use paste::paste;
    use tch::Tensor;

    #[test]
    fn test_tools() {
        assert_close!(1.0 + 2.0, 3.0);
        assert_close!(1e-3, 1e-3 + 1e-10);
        assert_not_close!(1e-3, 2e-3);
        assert_not_close!(1e-3, 1e-3 + 1e-5);
    }

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
        let (xmg, ymg) = (x.clone(), y);

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
        let (xmg, ymg) = (x.clone(), y);

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
        let g = &g + 10.0 / f.clone();
        g.backward();
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
        g.backward();
        let (apt, bpt, gpt) = (a, b, g);

        // forward pass went well
        assert_close!(gmg.data(), gpt.double_value(&[]));

        // backward pass went well
        // TODO - note that even strict equality is working, indicating probably op-for-op equivalence
        // When we would be satisfied with merely achieving assert_close
        assert_eq!(amg.grad().unwrap(), apt.grad().double_value(&[]));
        assert_eq!(bmg.grad().unwrap(), bpt.grad().double_value(&[]));
    }
}
