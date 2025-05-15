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
    pub backward_fn: Option<fn() -> f64>,
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
    pub fn new(data: f64, grad: Option<f64>, backward_fn: Option<fn() -> f64>, prev_nodes: Option<Vec<Value>>) -> Self {
        ValueInner { data, grad, backward_fn, prev_nodes }
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
    pub fn new(data: f64, grad: Option<f64>, backward_fn: Option<fn() -> f64>, prev_nodes: Option<Vec<Value>>) -> Self {
        Value(Rc::new(RefCell::new(ValueInner::new(data, grad, backward_fn, prev_nodes))))
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

    // pub fn pow(&self, n: f64) -> Value {
    //     Value::new(self.data().powf(n))
    //     /*

    //     def __pow__(self, other):
    //         assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    //         out = Value(self.data**other, (self,), f'**{other}')

    //         def _backward():
    //             self.grad += (other * self.data**(other-1)) * out.grad
    //         out._backward = _backward

    //         return out

    //      */
    // }
    pub fn backwards(&self) {
        // Topological order means for all directed edges  parent->child, parent appears first
        // To easily satisfy this property, we add each child, then add its parents, and reverse the whole list at the end

        let visited = &mut HashSet::new(); // For quick lookup
        let topo_rev = &mut Vec::new(); // The actual topological order

        // // start the recursion
        build_topo(&self, visited, topo_rev);

        self.borrow_mut().grad = Some(1.0);
        for v in topo_rev.iter().rev() {
            let mut v = v.borrow_mut();
            match v.backward_fn {
                None => {}
                Some(f) => v.grad = Some(f()),
            }
        }
    }

    pub fn relu(&self) -> Value {
        Value::from(if self.data() < 0.0 { 0.0 } else { self.data() })
    }
}

macro_rules! impl_binary_op {
    ($trait:ident, $method:ident, $operator:tt) =>
    (
        // Implement where Value is on LHS or both operands are Value
        impl<T> $trait<T> for Value
        where T: Into<Value>,
        {
            type Output = Self;
            fn $method(self, rhs: T) -> Self {
                Value::from(self.data() $operator rhs.into().data())
            }
        }

        // Implement where Value is on RHS
        impl $trait<Value> for f64 {
            type Output = Value;
            fn $method(self, rhs: Value) -> Value {
                Value::from(self $operator rhs.data())
            }
        }
    )
}

impl_binary_op!(Add, add, +);
impl_binary_op!(Mul, mul, *);
impl_binary_op!(Div, div, /);
impl_binary_op!(Sub, sub, -);
// TODO - also impl +=, -=, etc
