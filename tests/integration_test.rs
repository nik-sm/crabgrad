#[cfg(test)]
mod tests {
    use micrograd_rs::engine::Value;
    use micrograd_rs::nn::MLP;
    use micrograd_rs::nn::Trainer;
    use micrograd_rs::nn::cross_entropy;
    use micrograd_rs::{assert_close, assert_not_close};
    use paste::paste;
    use rand;

    use rand_distr::{Distribution, Normal};

    fn fake_data() -> (Vec<f64>, Vec<u8>) {
        let n = 1000;
        let rng = &mut rand::thread_rng();
        let mut data: Vec<f64> = Vec::with_capacity(2 * n);
        let mut labels: Vec<u8> = Vec::with_capacity(2 * n);

        // class 0
        let normal = Normal::new(0.0, 1.0).unwrap();
        data.extend((0..n).map(|_| normal.sample(rng)));
        labels.extend(vec![0u8; n]);

        // class 1
        let normal = Normal::new(2.0, 1.0).unwrap();
        data.extend((0..n).map(|_| normal.sample(rng)));
        labels.extend(vec![0u8; n]);

        (data, labels)
    }

    #[test]
    fn test_tools() {
        assert_close!(1.0 + 2.0, 3.0);
        assert_close!(1e-3, 1e-3 + 1e-10);
        assert_not_close!(1e-3, 2e-3);
        assert_not_close!(1e-3, 1e-3 + 1e-5);
    }

    /// Test the use of an operator on a Value, from either side
    macro_rules! test_op {
        ($val1:expr, $val2:expr, $name:ident, $op:tt, $result:expr) => {
            paste! {
            #[test]
                fn [<test_ $name _both>]() {
                    assert_close!((Value::new($val1) $op Value::new($val2)).borrow().data, $result)
                }
            }

            paste! {
            #[test]
                fn [<test_ $name _lhs>] () {
                    let res = ($val1 $op Value::new($val2));
                    let b = res.borrow();
                    assert_close!(b.data, $result)
                }
            }

            paste! {
                #[test]
                fn [<test_ $name _rhs>] () {
                    assert_close!((Value::new($val1) $op $val2).borrow().data, $result)
                }
            }
        };
    }

    test_op!(5.0, 6.0, add, +, 11.0);
    test_op!(5.0, 6.0, sub, -, -1.0);
    test_op!(5.0, 6.0, mul, *, 30.0);
    test_op!(5.0, 6.0, div, /, 5./6.);

    // #[test]
    // fn test_mlp() {
    //     let (data, labels) = fake_data();
    //     let mlp = MLP::new();
    //     let trainer = Trainer::new(mlp, None).fit(&data, &labels);

    //     let logits = mlp.forward(data);
    //     let loss = cross_entropy(&labels, &logits);
    // }

    #[test]
    fn verify_hashset_and_eq_behavior_for_rc() {
        use std::collections::HashSet;
        use std::rc::Rc;
        let mut table = HashSet::new();
        let item1 = Rc::new(Value::new(5.0));
        let item1_clone = Rc::clone(&item1);

        let item2 = Rc::new(Value::new(5.0));
        table.insert(Rc::clone(&item1));

        assert!(table.contains(&item1));
        assert!(table.contains(&item1_clone));

        assert!(!table.contains(&item2));

        assert!(item1 == item1);
        assert!(item1 == item1_clone);
        assert!(item1 != item2);
    }

    #[test]
    fn verify_clone_behavior() {
        // Imagining that we create a node at one stage as an Rc<..>,
        // then later need to mutate it during backward (changing its grad attribute)
        use micrograd_rs::{Value, engine::ValueInner};
        use std::cell::RefCell;
        use std::collections::HashSet;
        use std::rc::Rc;

        // Create a graph with a cycle and see whether hashset with cloned items instead of refs is OK for detecting previosly visited nodes
        let item1 = Value::new(1.23);
        let item2 = Value(Rc::new(RefCell::new(ValueInner {
            data: 7.89,
            grad: None,
            backward_fn: None,
            prev_nodes: Some(vec![item1.clone()]),
        })));

        let mut visited = HashSet::<&Value>::new();
        visited.insert(&item1);

        assert!(visited.contains(&item1));
        if let Some(prev) = &item2.borrow().prev_nodes {
            dbg!("check prevs");
            for child in prev {
                assert!(visited.contains(&child));
            }
        }
    }
}
