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
    use tch::Tensor;

    fn fake_data() -> (Vec<f64>, Vec<u8>) {
        let n = 1000;
        let mut rng = rand::rng();
        let mut data: Vec<f64> = Vec::with_capacity(2 * n);
        let mut labels: Vec<u8> = Vec::with_capacity(2 * n);

        // class 0
        let normal = Normal::new(0.0, 1.0).unwrap();

        data.extend(normal.sample_iter(&mut rng).take(n));
        labels.extend(vec![0u8; n]);

        // class 1
        let normal = Normal::new(2.0, 1.0).unwrap();
        data.extend(normal.sample_iter(&mut rng).take(n));
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

    // #[test]
    // fn test_mlp() {
    //     let (data, labels) = fake_data();
    //     let mlp = MLP::new();
    //     let trainer = Trainer::new(mlp, None).fit(&data, &labels);

    //     let logits = mlp.forward(data);
    //     let loss = cross_entropy(&labels, &logits);
    // }

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
    fn check_torch_install() {
        let t = Tensor::from_slice(&[3, 1, 4, 1, 5]);
        let t = t * 2;
        t.print();
    }

    #[test]
    fn compare_to_torch() {
        let x = Value::from(-4.0);
        let z = 2 * x + 2 + x;
        let q = z.relu() + z * x;
        let h = (z * z).relu();
        let y = h + q + q * x;
        y.backward();
        let (xmg, ymg) = (x, y);

        let mut x = Tensor::from(-4.0);
        x.set_requires_grad(true);
        let z: Tensor = 2.0 * x + 2.0 + x;
        let q = z.relu() + z * x;
        let h = (z * z).relu();
        let y = h + q + q * x;
        y.backward();
        let (xpt, ypt) = (x, y);

        //  forward pass went well
        assert_eq!(ymg.data(), ypt.data().double_value(&[0]));
        // backward pass went well
        assert_eq!(xmg.grad().unwrap(), xpt.grad().double_value(&[0]));

        // def test_more_ops():

        //     a = Value(-4.0)
        //     b = Value(2.0)
        //     c = a + b
        //     d = a * b + b**3
        //     c += c + 1
        //     c += 1 + c + (-a)
        //     d += d * 2 + (b + a).relu()
        //     d += 3 * d + (b - a).relu()
        //     e = c - d
        //     f = e**2
        //     g = f / 2.0
        //     g += 10.0 / f
        //     g.backward()
        //     amg, bmg, gmg = a, b, g

        //     a = torch.Tensor([-4.0]).double()
        //     b = torch.Tensor([2.0]).double()
        //     a.requires_grad = True
        //     b.requires_grad = True
        //     c = a + b
        //     d = a * b + b**3
        //     c = c + c + 1
        //     c = c + 1 + c + (-a)
        //     d = d + d * 2 + (b + a).relu()
        //     d = d + 3 * d + (b - a).relu()
        //     e = c - d
        //     f = e**2
        //     g = f / 2.0
        //     g = g + 10.0 / f
        //     g.backward()
        //     apt, bpt, gpt = a, b, g

        //     tol = 1e-6
        //     # forward pass went well
        //     assert abs(gmg.data - gpt.data.item()) < tol
        //     # backward pass went well
        //     assert abs(amg.grad - apt.grad.item()) < tol
        //     assert abs(bmg.grad - bpt.grad.item()) < tol
    }
}
