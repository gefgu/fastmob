use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
pub(crate) fn canonical_adjacency_form(n_nodes: u32, edges: Vec<(u32, u32)>) -> PyResult<String> {
    if n_nodes > 11 {
        return Err(PyValueError::new_err(
            "canonical_adjacency_form supports at most 11 nodes (11! permutations)",
        ));
    }

    let n = n_nodes as usize;
    let bits = n * n;

    let mut adj = vec![false; n * n];
    for (u, v) in &edges {
        let u = *u as usize;
        let v = *v as usize;
        if u >= n || v >= n {
            return Err(PyValueError::new_err(format!(
                "edge ({}, {}) references node index >= n_nodes ({})",
                u, v, n
            )));
        }
        adj[u * n + v] = true;
    }

    let mut perm: Vec<usize> = (0..n).collect();
    let mut best: u128 = 0;

    loop {
        let mut val: u128 = 0;
        for i in 0..n {
            for j in 0..n {
                val <<= 1;
                if adj[perm[i] * n + perm[j]] {
                    val |= 1;
                }
            }
        }
        if val > best {
            best = val;
        }

        if !next_permutation(&mut perm) {
            break;
        }
    }

    Ok(format!("{:0>width$b}", best, width = bits))
}

fn next_permutation(v: &mut [usize]) -> bool {
    let n = v.len();
    if n < 2 {
        return false;
    }

    let mut i = n - 1;
    loop {
        if i == 0 {
            return false;
        }
        i -= 1;
        if v[i] < v[i + 1] {
            break;
        }
    }

    let mut j = n - 1;
    while v[j] <= v[i] {
        j -= 1;
    }
    v.swap(i, j);
    v[i + 1..].reverse();
    true
}
