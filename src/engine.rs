use std::cell::RefCell;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::ops::{Add, Deref, Div, Mul, Sub};
use std::ptr;
use std::rc::Rc;

const PRECISION: usize = 3;

type BackwardType2 = fn(Value, Value) -> Value;

#[derive(Debug, Clone)]
pub struct ValueInner {
    pub data: f64,
    pub grad: Option<f64>,
    pub backward_fn: Option<BackwardType2>,
    pub prev_nodes: Option<Vec<Value>>,
}

#[derive(Debug, Clone)]
pub struct Value(pub Rc<RefCell<ValueInner>>); // TODO - remove pub after testing

impl From<f64> for Value {
    fn from(data: f64) -> Self {
        Self::new(data)
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self, other)
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self as *const Self).hash(state);
    }
}

impl ValueInner {
    pub fn new(data: f64) -> Self {
        return ValueInner { data, grad: None, backward_fn: None, prev_nodes: None };
    }
}

impl Deref for Value {
    type Target = Rc<RefCell<ValueInner>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Value {
    pub fn new(data: f64) -> Self {
        Value(Rc::new(RefCell::new(ValueInner::new(data))))
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

    pub fn pow(&self, n: f64) -> Value {
        Value::new(self.data().powf(n))
        /*

        def __pow__(self, other):
            assert isinstance(other, (int, float)), "only supporting int/float powers for now"
            out = Value(self.data**other, (self,), f'**{other}')

            def _backward():
                self.grad += (other * self.data**(other-1)) * out.grad
            out._backward = _backward

            return out

         */
    }
    pub fn backwards(&self) {
        // Topological order means for all directed edges  parent->child, parent appears first
        // To easily satisfy this property, we add each child, then add its parents, and reverse the whole list at the end

        let mut visited = HashSet::<&Value>::new(); // For quick lookup
        let mut topo = Vec::<&Value>::new(); // The actual topological order

        // // start the recursion
        self.build_topo(&self, &mut visited, &mut topo);

        self.borrow_mut().grad = Some(1.0);
        // for v in topo.iter().rev() {
        //     todo!();
        // }
    }

    fn build_topo<'a>(&'a self, node: &'a Value, visited: &mut HashSet<&'a Value>, topo: &mut Vec<&'a Value>) -> () {
        // v is the child node and we have a link to our parent nodes
        todo!();
        if visited.insert(node) {
            // Be sure our parent nodes are visited and added to the (reversed) topo
            match &node.borrow().prev_nodes {
                Some(prev_nodes) => prev_nodes.iter().for_each(|parent| self.build_topo(parent, visited, topo)),
                None => {}
            }
            // if let Some(prev) = &node.borrow().prev_nodes {
            //     for child in prev.iter() {
            //         self.build_topo(child, visited, topo);
            //     }
            // }

            // Add us (the child) to the topo
            topo.push(node);
        }
    }

    pub fn relu(&self) -> Value {
        Value::new(if self.data() < 0.0 { 0.0 } else { self.data() })
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
                Value::new(self.data() $operator rhs.into().data())
            }
        }

        // Implement where Value is on RHS
        impl $trait<Value> for f64 {
            type Output = Value;
            fn $method(self, rhs: Value) -> Value {
                Value::new(self $operator rhs.data())
            }
        }
    )
}

impl_binary_op!(Add, add, +);
impl_binary_op!(Mul, mul, *);
impl_binary_op!(Div, div, /);
impl_binary_op!(Sub, sub, -);
// TODO - also impl +=, -=, etc
