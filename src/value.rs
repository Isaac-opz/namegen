use smallvec::SmallVec;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Value(pub u32);

#[derive(Debug, Clone)]
pub enum Op {
    Input,
    Add(Value, Value),
    Mul(Value, Value),
    Pow(Value, f64),
    Exp(Value),
    Log(Value),
    ReLU(Value),
    Sum(SmallVec<[Value; 32]>), // increased to 32 for vocab_size
    Dot(SmallVec<[Value; 16]>, SmallVec<[Value; 16]>),
}

#[derive(Debug, Clone)]
pub struct Node {
    pub data: f64,
    pub grad: f64,
    pub op: Op,
}

use parking_lot::Mutex;
use std::sync::Arc;

#[derive(Debug, Clone, Default)]
struct GraphInner {
    pub nodes: Vec<Node>,
}

#[derive(Debug, Clone, Default)]
pub struct Graph(Arc<Mutex<GraphInner>>);

impl Graph {
    pub fn new() -> Self {
        Self::with_arena_capacity(16384)
    }

    pub fn with_arena_capacity(capacity: usize) -> Self {
        Self(Arc::new(Mutex::new(GraphInner {
            nodes: Vec::with_capacity(capacity),
        })))
    }

    pub fn reserve_arena(&self, additional: usize) {
        self.0.lock().nodes.reserve(additional);
    }

    pub fn value(&self, data: f64) -> Value {
        let mut inner = self.0.lock();
        let id = inner.nodes.len() as u32;
        inner.nodes.push(Node {
            data,
            grad: 0.0,
            op: Op::Input,
        });
        Value(id)
    }

    pub fn add(&self, a: Value, b: Value) -> Value {
        let mut inner = self.0.lock();
        let data = inner.nodes[a.0 as usize].data + inner.nodes[b.0 as usize].data;
        let id = inner.nodes.len() as u32;
        inner.nodes.push(Node {
            data,
            grad: 0.0,
            op: Op::Add(a, b),
        });
        Value(id)
    }

    pub fn mul(&self, a: Value, b: Value) -> Value {
        let mut inner = self.0.lock();
        let data = inner.nodes[a.0 as usize].data * inner.nodes[b.0 as usize].data;
        let id = inner.nodes.len() as u32;
        inner.nodes.push(Node {
            data,
            grad: 0.0,
            op: Op::Mul(a, b),
        });
        Value(id)
    }

    pub fn pow(&self, a: Value, p: f64) -> Value {
        let mut inner = self.0.lock();
        let data = inner.nodes[a.0 as usize].data.powf(p);
        let id = inner.nodes.len() as u32;
        inner.nodes.push(Node {
            data,
            grad: 0.0,
            op: Op::Pow(a, p),
        });
        Value(id)
    }

    pub fn exp(&self, a: Value) -> Value {
        let mut inner = self.0.lock();
        let data = inner.nodes[a.0 as usize].data.exp();
        let id = inner.nodes.len() as u32;
        inner.nodes.push(Node {
            data,
            grad: 0.0,
            op: Op::Exp(a),
        });
        Value(id)
    }

    pub fn log(&self, a: Value) -> Value {
        let mut inner = self.0.lock();
        let data = inner.nodes[a.0 as usize].data.ln();
        let id = inner.nodes.len() as u32;
        inner.nodes.push(Node {
            data,
            grad: 0.0,
            op: Op::Log(a),
        });
        Value(id)
    }

    pub fn relu(&self, a: Value) -> Value {
        let mut inner = self.0.lock();
        let data = inner.nodes[a.0 as usize].data.max(0.0);
        let id = inner.nodes.len() as u32;
        inner.nodes.push(Node {
            data,
            grad: 0.0,
            op: Op::ReLU(a),
        });
        Value(id)
    }

    pub fn sum(&self, vals: &[Value]) -> Value {
        let mut inner = self.0.lock();
        let data = vals.iter().map(|&v| inner.nodes[v.0 as usize].data).sum();
        let id = inner.nodes.len() as u32;
        inner.nodes.push(Node {
            data,
            grad: 0.0,
            op: Op::Sum(SmallVec::from_slice(vals)),
        });
        Value(id)
    }

    pub fn dot(&self, a: &[Value], b: &[Value]) -> Value {
        use wide::f64x2;
        let mut inner = self.0.lock();

        let nodes_ptr = inner.nodes.as_ptr();
        let mut sum = 0.0;
        let n = a.len();
        let mut i = 0;

        while i + 1 < n {
            unsafe {
                let va = f64x2::from([
                    (*nodes_ptr.add(a[i].0 as usize)).data,
                    (*nodes_ptr.add(a[i + 1].0 as usize)).data,
                ]);
                let vb = f64x2::from([
                    (*nodes_ptr.add(b[i].0 as usize)).data,
                    (*nodes_ptr.add(b[i + 1].0 as usize)).data,
                ]);
                let prod = va * vb;
                sum += prod.reduce_add();
            }
            i += 2;
        }

        while i < n {
            unsafe {
                sum +=
                    (*nodes_ptr.add(a[i].0 as usize)).data * (*nodes_ptr.add(b[i].0 as usize)).data;
            }
            i += 1;
        }

        let id = inner.nodes.len() as u32;
        inner.nodes.push(Node {
            data: sum,
            grad: 0.0,
            op: Op::Dot(SmallVec::from_slice(a), SmallVec::from_slice(b)),
        });
        Value(id)
    }

    pub fn backward(&self, root: Value) {
        let mut inner = self.0.lock();
        let nodes_count = inner.nodes.len();
        if nodes_count == 0 {
            return;
        }

        // Reset all grads
        for node in &mut inner.nodes {
            node.grad = 0.0;
        }
        inner.nodes[root.0 as usize].grad = 1.0;

        // Use raw pointer for faster, unsafe access to bypass borrow checker and avoid clones
        let nodes_ptr = inner.nodes.as_mut_ptr();

        for i in (0..nodes_count).rev() {
            unsafe {
                let node_ptr = nodes_ptr.add(i);
                let grad = (*node_ptr).grad;
                if grad == 0.0 {
                    continue;
                }

                // Note: we can safely borrow Op because we are accessing previous nodes for gradients
                match &(*node_ptr).op {
                    Op::Input => {}
                    Op::Add(a, b) => {
                        (*nodes_ptr.add(a.0 as usize)).grad += grad;
                        (*nodes_ptr.add(b.0 as usize)).grad += grad;
                    }
                    Op::Mul(a, b) => {
                        let da = (*nodes_ptr.add(a.0 as usize)).data;
                        let db = (*nodes_ptr.add(b.0 as usize)).data;
                        (*nodes_ptr.add(a.0 as usize)).grad += db * grad;
                        (*nodes_ptr.add(b.0 as usize)).grad += da * grad;
                    }
                    Op::Pow(a, p) => {
                        let da = (*nodes_ptr.add(a.0 as usize)).data;
                        (*nodes_ptr.add(a.0 as usize)).grad += p * da.powf(p - 1.0) * grad;
                    }
                    Op::Exp(a) => {
                        let d = (*node_ptr).data;
                        (*nodes_ptr.add(a.0 as usize)).grad += d * grad;
                    }
                    Op::Log(a) => {
                        let da = (*nodes_ptr.add(a.0 as usize)).data;
                        (*nodes_ptr.add(a.0 as usize)).grad += (1.0 / da) * grad;
                    }
                    Op::ReLU(a) => {
                        let da = (*nodes_ptr.add(a.0 as usize)).data;
                        if da > 0.0 {
                            (*nodes_ptr.add(a.0 as usize)).grad += grad;
                        }
                    }
                    Op::Sum(vals) => {
                        for v in vals {
                            (*nodes_ptr.add(v.0 as usize)).grad += grad;
                        }
                    }
                    Op::Dot(a, b) => {
                        for (&ai, &bi) in a.iter().zip(b.iter()) {
                            let da = (*nodes_ptr.add(ai.0 as usize)).data;
                            let db = (*nodes_ptr.add(bi.0 as usize)).data;
                            (*nodes_ptr.add(ai.0 as usize)).grad += db * grad;
                            (*nodes_ptr.add(bi.0 as usize)).grad += da * grad;
                        }
                    }
                }
            }
        }
    }

    pub fn truncate(&self, count: usize) {
        let mut inner = self.0.lock();
        inner.nodes.truncate(count);
    }

    pub fn nodes_count(&self) -> usize {
        self.0.lock().nodes.len()
    }

    pub fn node_data(&self, id: Value) -> f64 {
        self.0.lock().nodes[id.0 as usize].data
    }

    pub fn add_node_data(&self, id: Value, delta: f64) {
        self.0.lock().nodes[id.0 as usize].data += delta;
    }

    pub fn node_grad(&self, id: Value) -> f64 {
        self.0.lock().nodes[id.0 as usize].grad
    }

    pub fn softmax(&self, logits: &[Value]) -> Vec<Value> {
        let inner = self.0.lock();
        let max_val = logits
            .iter()
            .map(|&v| inner.nodes[v.0 as usize].data)
            .fold(f64::NEG_INFINITY, f64::max);

        drop(inner);

        let max_node = self.value(max_val);
        let mut exp_nodes = Vec::with_capacity(logits.len());
        for &v in logits {
            let diff = self.sub(v, max_node);
            exp_nodes.push(self.exp(diff));
        }
        let total = self.sum(&exp_nodes);
        exp_nodes.into_iter().map(|e| self.div(e, total)).collect()
    }

    pub fn rmsnorm(&self, x: &[Value]) -> Vec<Value> {
        let n = x.len() as f64;
        let dot = self.dot(x, x);
        let ms = self.div(dot, self.value(n));
        let scale = self.pow(self.add(ms, self.value(1e-5)), -0.5);
        x.iter().map(|&xi| self.mul(xi, scale)).collect()
    }

    pub fn sub(&self, a: Value, b: Value) -> Value {
        let neg_b = self.mul(b, self.value(-1.0));
        self.add(a, neg_b)
    }

    pub fn div(&self, a: Value, b: Value) -> Value {
        let inv_b = self.pow(b, -1.0);
        self.mul(a, inv_b)
    }
}
