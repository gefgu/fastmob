use std::f64::consts::TAU;
use std::fs;
use std::path::{Path, PathBuf};

pub const ADJACENCY_SHIFT_BITS: u32 = 36;
const ADJACENCY_MASK: u64 = (1_u64 << ADJACENCY_SHIFT_BITS) - 1;
const VIEWBOX_SIZE: f64 = 100.0;
const NODE_RADIUS: f64 = 5.5;

pub const LITERATURE_MOTIF_IDS: [u64; 17] = [
    0x1000000000,
    0x2000000006,
    0x30000000e4,
    0x300000008c,
    0x30000000e2,
    0x4000006818,
    0x4000004218,
    0x4000007888,
    0x4000006984,
    0x5000c80830,
    0x5000820830,
    0x5000e84030,
    0x5000c10610,
    0x6620102060,
    0x6408102060,
    0x6720802060,
    0x66040a0060,
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Motif {
    pub node_count: usize,
    pub edges: Vec<(usize, usize)>,
}

pub fn decode_packed_motif_id(motif_id: u64) -> Result<Motif, String> {
    let node_count = (motif_id >> ADJACENCY_SHIFT_BITS) as usize;
    let adjacency = motif_id & ADJACENCY_MASK;
    if node_count == 0 || node_count > 6 {
        return Err(format!(
            "decoded node count must be between 1 and 6, got {node_count}"
        ));
    }

    let used_bits = node_count * node_count;
    if adjacency >= (1_u64 << used_bits) {
        return Err("adjacency contains bits outside the node matrix".to_string());
    }

    let mut edges = Vec::new();
    for source in 0..node_count {
        for target in 0..node_count {
            let bit_index = used_bits - 1 - (source * node_count + target);
            if (adjacency >> bit_index) & 1 == 1 {
                edges.push((source, target));
            }
        }
    }
    Ok(Motif { node_count, edges })
}

pub fn start_node_is_in_cycle(motif: &Motif) -> bool {
    motif.edges.iter().any(|edge| *edge == (0, 0))
        || motif
            .edges
            .iter()
            .filter(|(source, _)| *source == 0)
            .any(|(_, target)| can_reach(motif, *target, 0, &mut vec![false; motif.node_count]))
}

fn can_reach(motif: &Motif, current: usize, destination: usize, visited: &mut [bool]) -> bool {
    if current == destination {
        return true;
    }
    if visited[current] {
        return false;
    }
    visited[current] = true;
    motif
        .edges
        .iter()
        .filter(|(source, _)| *source == current)
        .any(|(_, target)| can_reach(motif, *target, destination, visited))
}

fn node_positions(node_count: usize) -> Vec<(f64, f64)> {
    if node_count == 1 {
        return vec![(50.0, 50.0)];
    }
    (0..node_count)
        .map(|index| {
            let angle = -TAU / 4.0 + TAU * index as f64 / node_count as f64;
            (50.0 + 31.0 * angle.cos(), 50.0 + 31.0 * angle.sin())
        })
        .collect()
}

fn shortened_line(start: (f64, f64), end: (f64, f64)) -> ((f64, f64), (f64, f64)) {
    let dx = end.0 - start.0;
    let dy = end.1 - start.1;
    let length = dx.hypot(dy);
    let ux = dx / length;
    let uy = dy / length;
    let margin = NODE_RADIUS + 1.5;
    (
        (start.0 + ux * margin, start.1 + uy * margin),
        (end.0 - ux * margin, end.1 - uy * margin),
    )
}

fn edge_path(source: usize, target: usize, positions: &[(f64, f64)], reciprocal: bool) -> String {
    let start = positions[source];
    if source == target {
        return format!(
            "M {:.2} {:.2} C {:.2} {:.2}, {:.2} {:.2}, {:.2} {:.2}",
            start.0 - 3.5,
            start.1 - 4.5,
            start.0 - 18.0,
            start.1 - 23.0,
            start.0 + 18.0,
            start.1 - 23.0,
            start.0 + 3.5,
            start.1 - 4.5,
        );
    }

    let (from, to) = shortened_line(start, positions[target]);
    if !reciprocal {
        return format!("M {:.2} {:.2} L {:.2} {:.2}", from.0, from.1, to.0, to.1);
    }

    let dx = to.0 - from.0;
    let dy = to.1 - from.1;
    let length = dx.hypot(dy);
    let control = (
        (from.0 + to.0) / 2.0 - dy / length * 9.0,
        (from.1 + to.1) / 2.0 + dx / length * 9.0,
    );
    format!(
        "M {:.2} {:.2} Q {:.2} {:.2} {:.2} {:.2}",
        from.0, from.1, control.0, control.1, to.0, to.1
    )
}

pub fn render_svg(motif_id: u64) -> Result<String, String> {
    let motif = decode_packed_motif_id(motif_id)?;
    let positions = node_positions(motif.node_count);
    let mut output = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {VIEWBOX_SIZE:.0} {VIEWBOX_SIZE:.0}\" role=\"img\" aria-label=\"Mobility motif {motif_id:#x}\">\n"
    );
    output.push_str(
        "  <defs><marker id=\"arrow\" viewBox=\"0 0 10 10\" refX=\"8\" refY=\"5\" markerWidth=\"5\" markerHeight=\"5\" orient=\"auto-start-reverse\"><path d=\"M 0 0 L 10 5 L 0 10 z\" fill=\"#111111\"/></marker></defs>\n",
    );
    output.push_str("  <g fill=\"none\" stroke=\"#111111\" stroke-width=\"1.8\" stroke-linecap=\"round\" marker-end=\"url(#arrow)\">\n");
    for &(source, target) in &motif.edges {
        let reciprocal = source != target && motif.edges.contains(&(target, source));
        output.push_str(&format!(
            "    <path d=\"{}\"/>\n",
            edge_path(source, target, &positions, reciprocal)
        ));
    }
    output.push_str("  </g>\n");
    let start_color = if start_node_is_in_cycle(&motif) {
        "#d62728"
    } else {
        "#111111"
    };
    for (index, &(x, y)) in positions.iter().enumerate() {
        let color = if index == 0 { start_color } else { "#111111" };
        output.push_str(&format!(
            "  <circle cx=\"{x:.2}\" cy=\"{y:.2}\" r=\"{NODE_RADIUS:.1}\" fill=\"{color}\"/>\n"
        ));
    }
    output.push_str("</svg>\n");
    Ok(output)
}

pub fn generate_literature_svgs(output_dir: &Path) -> Result<Vec<PathBuf>, String> {
    fs::create_dir_all(output_dir)
        .map_err(|error| format!("failed to create {}: {error}", output_dir.display()))?;
    let mut paths = Vec::with_capacity(LITERATURE_MOTIF_IDS.len());
    for motif_id in LITERATURE_MOTIF_IDS {
        let path = output_dir.join(format!("{motif_id:#x}.svg"));
        fs::write(&path, render_svg(motif_id)?)
            .map_err(|error| format!("failed to write {}: {error}", path.display()))?;
        paths.push(path);
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_adjacency_in_row_major_bit_order() {
        let motif = decode_packed_motif_id(0x2000000006).unwrap();
        assert_eq!(motif.node_count, 2);
        assert_eq!(motif.edges, vec![(0, 1), (1, 0)]);
    }

    #[test]
    fn rejects_invalid_node_counts_and_unused_bits() {
        assert!(decode_packed_motif_id(0).is_err());
        assert!(decode_packed_motif_id(0x7000000000).is_err());
        assert!(decode_packed_motif_id(0x1000000002).is_err());
    }

    #[test]
    fn detects_when_start_node_is_in_a_cycle() {
        let cycle = decode_packed_motif_id(0x2000000006).unwrap();
        let no_cycle = decode_packed_motif_id(0x1000000000).unwrap();
        assert!(start_node_is_in_cycle(&cycle));
        assert!(!start_node_is_in_cycle(&no_cycle));
    }

    #[test]
    fn svg_output_is_deterministic_and_marks_cycle_start_red() {
        let first = render_svg(0x2000000006).unwrap();
        let second = render_svg(0x2000000006).unwrap();
        assert_eq!(first, second);
        assert!(first.contains("viewBox=\"0 0 100 100\""));
        assert!(first.contains("fill=\"#d62728\""));
        assert_eq!(first.matches("<circle ").count(), 2);
    }

    #[test]
    fn literature_assets_have_unique_expected_names() {
        let mut names = LITERATURE_MOTIF_IDS
            .iter()
            .map(|motif_id| format!("{motif_id:#x}.svg"))
            .collect::<Vec<_>>();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), 17);
        assert!(names.contains(&"0x2000000006.svg".to_string()));
    }
}
