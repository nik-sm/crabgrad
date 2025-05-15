/*
In order to debug and understand how to implement nodes properly to allow backward-mode autodiff,
this is a minimal exmple of a graph data structure where we will print a topological order of nodes.

Nodes themselves will just contain a label, and the only supported operation to spawn new child nodes
will be a binary operation (addition).

*/
use std::cell::RefCell;
use std::collections::HashSet;
use std::convert::From;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::ptr;
use std::rc::Rc;

#[derive(Eq, PartialEq, Clone)]
pub struct Node(pub Rc<RefCell<NodeInner>>);

impl Deref for Node {
    type Target = Rc<RefCell<NodeInner>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Hash for Node {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let ref_cell_addr = &***self;
        ptr::hash(ref_cell_addr, state)
    }
}

#[derive(Eq)]
pub struct NodeInner {
    pub label: String,
    pub backward_data: Option<usize>, // The data that gets filled during backward pass
    pub prev: Vec<Node>,
}

impl PartialEq for NodeInner {
    fn eq(&self, other: &Self) -> bool {
        self.label == other.label && self.backward_data == other.backward_data && self.prev == other.prev
    }
}

impl<T> From<T> for NodeInner
where
    T: Into<String>,
{
    fn from(label: T) -> Self {
        Self { label: label.into(), prev: vec![], backward_data: None }
    }
}

impl<T> From<T> for Node
where
    T: Into<String>,
{
    fn from(label: T) -> Self {
        Self(Rc::new(RefCell::new(NodeInner::from(label))))
    }
}

fn build_topo<'a>(node: &'a Node, visited: &mut HashSet<&'a Node>, topo_rev: &mut Vec<&'a Node>) {
    if visited.insert(node) {
        let prev = node.borrow().prev.clone();
        for child in prev {
            build_topo(&child, visited, topo_rev)
        }
        topo_rev.push(node);
    }
    todo!()
}

impl Node {
    fn link<T: Into<String>>(self, other: Node, label: T) -> Self {
        Node(Rc::new(RefCell::new(NodeInner { label: label.into(), backward_data: None, prev: vec![self, other] })))
    }

    fn backward(self) {
        let visited = &mut HashSet::new();
        let topo_rev = &mut Vec::new();

        build_topo(&self, visited, topo_rev);

        // let topo_strings = topo.iter().map(|node| node.0.borrow().label.clone()).collect();

        self.borrow_mut().backward_data = Some(0);
        for (i, node) in topo_rev.iter().rev().enumerate() {
            node.borrow_mut().backward_data = Some(i + 1)
        }
    }
}

fn main() {
    // First example - no cycles
    let node1 = Node::from("dog");
    let node2 = Node::from("cat");

    let node3 = node1.link(node2, "bear");

    let node4 = Node::from("pizza");
    let node5 = node4.link(node3, "fry");

    node5.backward();
}
