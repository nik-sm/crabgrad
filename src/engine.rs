use std::cell::RefCell;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::ops::{Add, Deref, Div, Mul, Sub};
use std::ptr;
use std::rc::Rc;

const PRECISION: usize = 3;

#[derive(Debug, Clone)]
pub struct ValueInner {
    pub data: f64,
    pub grad: Option<f64>,
    pub backward_fn: Option<fn(&ValueInner)>,
    pub prev_nodes: Option<Vec<Value>>,
}

#[derive(Debug)]
pub struct Value(Rc<RefCell<ValueInner>>);

impl Clone for Value {
    /// To make it super clear that value.clone() only increments Rc
    fn clone(&self) -> Self {
        Value(Rc::clone(&self.0))
    }
}

impl From<f64> for Value {
    fn from(data: f64) -> Self {
        Self(Rc::new(RefCell::new(ValueInner::from(data))))
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self.0.as_ptr(), other.0.as_ptr())
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        ptr::hash(self.0.as_ptr(), state);
    }
}

impl ValueInner {
    pub fn new(data: f64, prev_nodes: Option<Vec<Value>>, backward_fn: Option<fn(&ValueInner)>) -> Self {
        ValueInner { data, grad: None, prev_nodes, backward_fn }
    }
}
impl From<f64> for ValueInner {
    fn from(data: f64) -> Self {
        ValueInner { data, grad: None, backward_fn: None, prev_nodes: None }
    }
}

impl Deref for Value {
    type Target = Rc<RefCell<ValueInner>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

fn build_topo(node: &Value, visited: &mut HashSet<Value>, topo_rev: &mut Vec<Value>) -> () {
    // v is the child node and we have a link to our parent nodes
    if !visited.contains(&node) {
        // PartialEq and Hash both use address of ValueInner
        visited.insert(node.clone());
        if let Some(prev) = &node.borrow().prev_nodes {
            for ancestor in prev {
                build_topo(&ancestor, visited, topo_rev)
            }
        }
        topo_rev.push(node.clone())
    }
}

impl Value {
    pub fn new(data: f64, prev_nodes: Option<Vec<Value>>, backward_fn: Option<fn(&ValueInner)>) -> Self {
        Value(Rc::new(RefCell::new(ValueInner::new(data, prev_nodes, backward_fn))))
    }

    pub fn data(&self) -> f64 {
        self.borrow().data
    }

    pub fn to_string(&self) -> String {
        let grad = match self.borrow().grad {
            Some(x) => format!("{x:.PRECISION$}"),
            None => "None".to_string(),
        };
        format!("data: {:.PRECISION$}\ngrad: {:.PRECISION$}", self.data(), grad)
    }

    pub fn pow<T: Into<Value>>(&self, exponent: T) -> Value {
        let exponent = exponent.into();

        let data = self.data().powf(exponent.data());
        let prev_nodes = vec![self.clone(), exponent.clone()];
        let backward_fn = |our_value_inner: &ValueInner| {
            // TODO - left off here
            match our_value_inner.prev_nodes.as_deref() {
                Some([base, exponent]) => {
                    let mut base = base.borrow_mut();
                    let mut exponent = exponent.borrow_mut();
                    let our_grad = our_value_inner.grad.unwrap_or(0.0);
                    // deriv of f = a ^ b   w.r.t.   a
                    base.grad =
                        Some(base.grad.unwrap_or(0.0) + exponent.data * base.data.powf(exponent.data - 1.0) * our_grad);
                    // deriv of f = a ^ b   w.r.t.   b
                    exponent.grad =
                        Some(exponent.grad.unwrap_or(0.0) + base.data.powf(exponent.data) * base.data.ln() * our_grad);
                }
                _ => {
                    unreachable!("binary op must have two ancestors")
                }
            }
        };

        Value::new(data, Some(prev_nodes), Some(backward_fn))
    }

    pub fn backwards(&self) {
        // Topological order means for all directed edges  parent->child, parent appears first
        // To easily satisfy this property, we add each child, then add its parents, and reverse the whole list at the end

        let visited = &mut HashSet::new(); // For quick lookup
        let topo_rev = &mut Vec::new(); // The actual topological order

        // // start the recursion
        build_topo(&self, visited, topo_rev);

        self.borrow_mut().grad = Some(1.0);
        for v in topo_rev.iter().rev() {
            if let Some(backprop) = &v.borrow().backward_fn {
                backprop(&v.borrow());
            }
        }
    }

    pub fn relu(&self) -> Value {
        let data = if self.data() <= 0.0 { 0.0 } else { self.data() };
        let prev_nodes = vec![self.clone()];
        let backward_fn = |our_value_inner: &ValueInner| match our_value_inner.prev_nodes.as_deref() {
            Some([first]) => {
                let mut first = first.borrow_mut();
                let our_grad = our_value_inner.grad.unwrap_or(0.0);
                let multiplier = if first.data > 0.0 { 1.0 } else { 0.0 };
                first.grad = Some(first.grad.unwrap_or(0.0) + multiplier * our_grad);
            }
            _ => {
                unreachable!("relu must have one ancestors")
            }
        };
        Value::new(data, Some(prev_nodes), Some(backward_fn))
    }
}

macro_rules! impl_binary_op {
    ($self:ident, $rhs:ident, $trait:ident, $method:ident, $operator:tt, $body:tt) =>
    (

        // Value on both sides
        impl $trait for Value
        {
            type Output = Self;
            fn $method($self, $rhs: Self) -> Self {
                $body
            }
        }

        // Value on RHS
        impl $trait<Value> for f64 {
            type Output = Value;
            fn $method(self, rhs: Value) -> Value {
                Value::from(self) $operator rhs
            }
        }

        // Value on LHS
        impl $trait<f64> for Value {
            type Output = Self;
            fn $method(self, rhs: f64) -> Self {
                self $operator Value::from(rhs)
            }
        }

        // Method-call style
        impl Value {
            fn $method<T: Into<Value>>(self, rhs: T) -> Self {
                self $operator rhs.into()
            }
        }
    )
}

impl_binary_op!(self, rhs, Add, add, +, {
    let data = self.data() + rhs.data();
    let prev_nodes = vec![self.clone(), rhs.clone()];
    let backward_fn = |our_value_inner: &ValueInner| {
        match our_value_inner.prev_nodes.as_deref() {
            Some([first, second]) => {
                let mut first = first.borrow_mut();
                let mut second = second.borrow_mut();
                let our_grad = our_value_inner.grad.unwrap_or(0.0);
                first.grad = Some(first.grad.unwrap_or(0.0) + our_grad);
                second.grad = Some(second.grad.unwrap_or(0.0) + our_grad);
            },
            _ => {
                unreachable!("binary op must have two ancestors")
            }
        }
    };
    Value::new(data, Some(prev_nodes), Some(backward_fn))
}
);
impl_binary_op!(self, rhs, Mul, mul, *, {
    let data = self.data() * rhs.data();
    let prev_nodes = vec![self.clone(), rhs.clone()];
    let backward_fn = |our_value_inner: &ValueInner| match our_value_inner.prev_nodes.as_deref() {
        Some([first, second]) => {
            let mut first = first.borrow_mut();
            let mut second = second.borrow_mut();
            let our_grad = our_value_inner.grad.unwrap_or(0.0);
            first.grad = Some(first.grad.unwrap_or(0.0) + second.data * our_grad);
            second.grad = Some(second.grad.unwrap_or(0.0) + first.data * our_grad);
        }
        _ => {
            unreachable!("binary op must have two ancestors")
        }
    };
    Value::new(data, Some(prev_nodes), Some(backward_fn))
});
impl_binary_op!(self, rhs, Div, div, /, {
    self * rhs.pow(-1.0)
});
impl_binary_op!(self, rhs, Sub, sub, -, {
    self + (-1.0 * rhs)
});

// TODO - also impl +=, -=, etc
