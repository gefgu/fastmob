//! H3-cell connected-component clustering for recurring stop locations.

use std::cmp::Reverse;

use h3o::{CellIndex, Resolution};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use super::h3::{INVALID_CELL, batch_latlng_to_cells};

#[derive(Debug, Clone, Copy)]
struct CellStats {
    count: usize,
    first_row: usize,
}

#[derive(Debug)]
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
    }

    fn find(&mut self, node: usize) -> usize {
        if self.parent[node] != node {
            let root = self.find(self.parent[node]);
            self.parent[node] = root;
        }
        self.parent[node]
    }

    fn union(&mut self, left: usize, right: usize) {
        let mut left = self.find(left);
        let mut right = self.find(right);
        if left == right {
            return;
        }
        if self.rank[left] < self.rank[right] {
            std::mem::swap(&mut left, &mut right);
        }
        self.parent[right] = left;
        if self.rank[left] == self.rank[right] {
            self.rank[left] += 1;
        }
    }
}

/// Cluster one user's already-tessellated cells by active H3 cells and
/// immediate neighbors.
fn cluster_cell_range(
    cells: &[u64],
    first_row: usize,
    min_samples: usize,
) -> Result<Vec<i32>, String> {
    let mut cell_stats: FxHashMap<u64, CellStats> = FxHashMap::default();
    for (offset, &cell) in cells.iter().enumerate() {
        cell_stats
            .entry(cell)
            .and_modify(|stats| stats.count += 1)
            .or_insert(CellStats {
                count: 1,
                first_row: first_row + offset,
            });
    }

    let active_cells: Vec<u64> = cell_stats
        .iter()
        .filter_map(|(&cell, stats)| (stats.count >= min_samples).then_some(cell))
        .collect();
    let active_ids: FxHashMap<u64, usize> = active_cells
        .iter()
        .enumerate()
        .map(|(id, &cell)| (cell, id))
        .collect();
    let mut components = UnionFind::new(active_cells.len());
    for (&cell, &cell_id) in &active_ids {
        let index = CellIndex::try_from(cell)
            .map_err(|_| "internal error: generated an invalid H3 cell".to_string())?;
        for neighbor in index.grid_disk::<Vec<_>>(1) {
            if let Some(&neighbor_id) = active_ids.get(&u64::from(neighbor)) {
                components.union(cell_id, neighbor_id);
            }
        }
    }

    let mut component_stats: FxHashMap<usize, CellStats> = FxHashMap::default();
    for (&cell, &cell_id) in &active_ids {
        let root = components.find(cell_id);
        let stats = cell_stats[&cell];
        component_stats
            .entry(root)
            .and_modify(|total| {
                total.count += stats.count;
                total.first_row = total.first_row.min(stats.first_row);
            })
            .or_insert(stats);
    }
    let mut ranked: Vec<(usize, CellStats)> = component_stats.into_iter().collect();
    ranked.sort_by_key(|(_, stats)| (Reverse(stats.count), stats.first_row));
    let component_labels: FxHashMap<usize, i32> = ranked
        .into_iter()
        .enumerate()
        .map(|(label, (root, _))| (root, label as i32))
        .collect();

    let mut labels = vec![-1_i32; cells.len()];
    for (offset, &cell) in cells.iter().enumerate() {
        if let Some(&cell_id) = active_ids.get(&cell) {
            let root = components.find(cell_id);
            labels[offset] = component_labels[&root];
        }
    }
    Ok(labels)
}

/// Cluster each contiguous user range by active H3 cells and immediate
/// neighbors. Coordinate tessellation and independent user ranges run on the
/// Rayon pool; the Arrow binding releases the GIL around this call.
pub fn h3_connected_components(
    lats: &[f64],
    lngs: &[f64],
    group_ends: &[usize],
    resolution: Resolution,
    min_samples: usize,
) -> Result<Vec<i32>, String> {
    if lats.len() != lngs.len() {
        return Err("latitude and longitude arrays must have the same length".to_string());
    }
    if min_samples == 0 {
        return Err("min_samples must be at least 1".to_string());
    }
    if group_ends.last().copied().unwrap_or(0) != lats.len()
        || group_ends.windows(2).any(|w| w[0] > w[1])
    {
        return Err("group_ends must be sorted and end at the coordinate length".to_string());
    }

    let cells = batch_latlng_to_cells(lats, lngs, resolution, None);
    if let Some(row) = cells.iter().position(|&cell| cell == INVALID_CELL) {
        return Err(format!(
            "invalid latitude/longitude at row {row}: ({}, {})",
            lats[row], lngs[row]
        ));
    }

    let starts = std::iter::once(0)
        .chain(group_ends.iter().copied())
        .collect::<Vec<_>>();
    let labels_by_user: Result<Vec<Vec<i32>>, String> = starts[..starts.len() - 1]
        .par_iter()
        .zip(group_ends.par_iter())
        .map(|(&start, &end)| cluster_cell_range(&cells[start..end], start, min_samples))
        .collect();
    Ok(labels_by_user?.into_iter().flatten().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearby_points_merge_and_distant_points_do_not() {
        let labels = h3_connected_components(
            &[48.8566, 48.8567, 48.9066],
            &[2.3522, 2.3523, 2.3522],
            &[3],
            Resolution::Eleven,
            1,
        )
        .unwrap();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], 0);
        assert_eq!(labels[2], 1);
    }

    #[test]
    fn inactive_cells_are_noise() {
        let labels = h3_connected_components(
            &[48.8566, 51.5, 40.712],
            &[2.3522, -0.127, -74.006],
            &[3],
            Resolution::Eleven,
            2,
        )
        .unwrap();
        assert_eq!(labels, vec![-1, -1, -1]);
    }
}
