use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Build a CSR adjacency list from an undirected edge list.
/// Edges are symmetrised; self-loops are ignored.
pub fn edges_to_csr(srcs: &[usize], dsts: &[usize], n_nodes: usize) -> (Vec<usize>, Vec<usize>) {
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
    for (&s, &d) in srcs.iter().zip(dsts.iter()) {
        if s != d && s < n_nodes && d < n_nodes {
            adj[s].push(d);
            adj[d].push(s);
        }
    }
    // deduplicate
    for row in &mut adj {
        row.sort_unstable();
        row.dedup();
    }
    let mut neighbor_starts = Vec::with_capacity(n_nodes + 1);
    let mut neighbors = Vec::new();
    neighbor_starts.push(0usize);
    for row in &adj {
        neighbors.extend_from_slice(row);
        neighbor_starts.push(neighbors.len());
    }
    (neighbor_starts, neighbors)
}

/// Random Geometric Graph: place `n_nodes` points uniformly in [0,1]²
/// and connect pairs within Euclidean distance `radius`.
/// Returns CSR adjacency (simple, undirected, no self-loops).
pub fn random_geometric_graph(n_nodes: usize, radius: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    if n_nodes == 0 {
        return (vec![0usize], Vec::new());
    }
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let xs: Vec<f64> = (0..n_nodes).map(|_| rng.gen_range(0.0_f64..1.0)).collect();
    let ys: Vec<f64> = (0..n_nodes).map(|_| rng.gen_range(0.0_f64..1.0)).collect();
    let r2 = radius * radius;

    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
    for i in 0..n_nodes {
        for j in (i + 1)..n_nodes {
            let dx = xs[i] - xs[j];
            let dy = ys[i] - ys[j];
            if dx * dx + dy * dy <= r2 {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }

    let mut neighbor_starts = Vec::with_capacity(n_nodes + 1);
    let mut neighbors = Vec::new();
    neighbor_starts.push(0usize);
    for row in &adj {
        neighbors.extend_from_slice(row);
        neighbor_starts.push(neighbors.len());
    }
    (neighbor_starts, neighbors)
}
