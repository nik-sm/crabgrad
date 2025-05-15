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
use std::ptr;
use std::rc::Rc;

#[derive(Clone, Debug)]
pub struct Node(pub Rc<RefCell<NodeInner>>);

impl Hash for Node {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let addr = self.0.as_ptr();
        ptr::hash(addr, state)
    }
}
impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self.0.as_ptr(), other.0.as_ptr())
    }
}
impl Eq for Node {}

#[derive(Debug)]
pub struct NodeInner {
    pub label: String,
    pub backward_data: Option<usize>, // The data that gets filled during backward pass
    pub prev: Vec<Node>,
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

fn build_topo(node: &Node, visited: &mut HashSet<Node>, topo_rev: &mut Vec<Node>) {
    if !visited.contains(node) {
        visited.insert(node.clone());
        let prev = &node.0.borrow().prev;
        for child in prev {
            build_topo(&child, visited, topo_rev)
        }
        topo_rev.push(node.clone());
    }
}

impl Node {
    fn link<S: Into<String>>(self, other: &Node, label: S) -> Self {
        Node(Rc::new(RefCell::new(NodeInner {
            label: label.into(),
            backward_data: None,
            prev: vec![self, other.clone()],
        })))
    }

    fn backward(self) {
        let visited = &mut HashSet::new();
        let topo_rev = &mut Vec::new();

        build_topo(&self, visited, topo_rev);

        // let topo_strings = topo.iter().map(|node| node.0.borrow().label.clone()).collect();

        self.0.borrow_mut().backward_data = Some(0);
        dbg!("self {:?}", self);
        for (i, node) in topo_rev.iter().rev().enumerate() {
            node.0.borrow_mut().backward_data = Some(i + 1);
            dbg!("node {i} {:?}", node);
        }
    }
}

fn main() {
    // First example - no cycles
    let reused = Node::from("reused");

    let node1 = Node::from("whoami");
    let node2 = Node::from("grandparent 2");

    let node3 = node1.link(&node2, "grandparent 1");
    let node4 = node3.link(&reused, "joining");

    let node5 = Node::from("parent 2");
    let node6 = node5.link(&node4, "parent 1");

    let node7 = node6.link(&reused, "root");

    node7.backward();
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn check_hashset_behavior() {
        let foo = Node::from("foo");
        let mut visited = HashSet::new();

        assert!(!visited.contains(&foo));

        visited.insert(&foo);
        dbg!("after", &visited);

        assert!(visited.contains(&foo));

        let foo2 = Node::from("foo");
        assert!(!visited.contains(&foo2));
    }
}
