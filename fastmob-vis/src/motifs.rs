use serde::Serialize;

pub const OTHER_MOTIF_ID: &str = "other";

pub const LITERATURE_MOTIF_PERCENTAGES: &[(i64, f64)] = &[
    (1, 10.10),
    (2, 30.80),
    (3, 12.70),
    (4, 9.40),
    (5, 0.60),
    (6, 7.30),
    (7, 5.00),
    (8, 1.70),
    (9, 0.70),
    (10, 3.00),
    (11, 2.30),
    (12, 1.30),
    (13, 1.00),
    (14, 1.30),
    (15, 1.30),
    (16, 0.72),
    (17, 0.86),
];

pub const LITERATURE_TO_FASTMOB_MOTIF_ID: &[(i64, i64)] = &[
    (1, 0x1000000000),
    (2, 0x2000000006),
    (3, 0x30000000E4),
    (4, 0x300000008C),
    (5, 0x30000000E2),
    (6, 0x4000006818),
    (7, 0x4000004218),
    (8, 0x4000007888),
    (9, 0x4000006984),
    (10, 0x5000C80830),
    (11, 0x5000820830),
    (12, 0x5000E84030),
    (13, 0x5000C10610),
    (14, 0x6620102060),
    (15, 0x6408102060),
    (16, 0x6720802060),
    (17, 0x66040A0060),
];

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MotifBasisRow {
    pub literature_motif_id: MotifId,
    pub motif_id: MotifId,
    pub hex_id: String,
    pub percentage: f64,
    pub count: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum MotifId {
    Integer(i64),
    Other(String),
}

impl MotifId {
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            MotifId::Integer(value) => Some(*value),
            MotifId::Other(_) => None,
        }
    }
}

pub fn format_motif_hex_id(motif_id: i64) -> String {
    format!("{motif_id:#x}")
}

fn round2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

fn empty_basis() -> Vec<MotifBasisRow> {
    let mut rows: Vec<MotifBasisRow> = LITERATURE_TO_FASTMOB_MOTIF_ID
        .iter()
        .map(|&(literature_motif_id, fastmob_motif_id)| MotifBasisRow {
            literature_motif_id: MotifId::Integer(literature_motif_id),
            motif_id: MotifId::Integer(fastmob_motif_id),
            hex_id: format_motif_hex_id(fastmob_motif_id),
            percentage: 0.0,
            count: 0,
        })
        .collect();
    rows.push(MotifBasisRow {
        literature_motif_id: MotifId::Other(OTHER_MOTIF_ID.to_string()),
        motif_id: MotifId::Other(OTHER_MOTIF_ID.to_string()),
        hex_id: "Other".to_string(),
        percentage: 0.0,
        count: 0,
    });
    rows
}

pub fn literature_distribution_rows() -> Vec<MotifBasisRow> {
    let mut rows = empty_basis();
    let total: f64 = LITERATURE_MOTIF_PERCENTAGES.iter().map(|&(_, p)| p).sum();
    let n = rows.len();
    for (row, &(_, percentage)) in rows[..n - 1].iter_mut().zip(LITERATURE_MOTIF_PERCENTAGES) {
        row.percentage = percentage;
    }
    rows[n - 1].percentage = (100.0 - total).max(0.0);
    rows
}

pub trait MotifDistributionLike {
    fn motif_id(&self) -> i64;
    fn percentage(&self) -> f64;
    fn count(&self) -> i64;
}

pub fn map_motif_distribution_to_literature_basis<T: MotifDistributionLike>(
    distribution: &[T],
) -> Vec<MotifBasisRow> {
    let mut rows = empty_basis();
    let n = rows.len();
    let mut by_motif_id = std::collections::BTreeMap::<i64, usize>::new();
    for (idx, row) in rows[..n - 1].iter().enumerate() {
        if let Some(motif_id) = row.motif_id.as_i64() {
            by_motif_id.insert(motif_id, idx);
        }
    }
    let other_idx = n - 1;

    for row in distribution {
        let target_idx = by_motif_id
            .get(&row.motif_id())
            .copied()
            .unwrap_or(other_idx);
        let target = &mut rows[target_idx];
        target.percentage = round2(target.percentage + row.percentage());
        target.count += row.count();
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Row {
        motif_id: i64,
        percentage: f64,
        count: i64,
    }

    impl MotifDistributionLike for Row {
        fn motif_id(&self) -> i64 {
            self.motif_id
        }

        fn percentage(&self) -> f64 {
            self.percentage
        }

        fn count(&self) -> i64 {
            self.count
        }
    }

    #[test]
    fn literature_rows_match_reference_basis() {
        let rows = literature_distribution_rows();
        assert_eq!(rows.len(), 18);
        assert_eq!(rows[0].hex_id, "0x1000000000");
        assert_eq!(rows[0].literature_motif_id, MotifId::Integer(1));
        assert_eq!(rows[0].motif_id, MotifId::Integer(68719476736));
        assert!((rows[0].percentage - 10.1).abs() < 1e-9);
        assert_eq!(rows[16].hex_id, "0x66040a0060");
        assert!((rows[16].percentage - 0.86).abs() < 1e-9);
        let other = rows.last().unwrap();
        assert_eq!(other.hex_id, "Other");
        assert_eq!(
            other.literature_motif_id,
            MotifId::Other(OTHER_MOTIF_ID.to_string())
        );
        assert!((other.percentage - 9.920000000000002).abs() < 1e-12);
    }

    #[test]
    fn maps_distribution_to_reference_basis() {
        let distribution = vec![
            Row {
                motif_id: 0x1000000000,
                count: 5,
                percentage: 50.0,
            },
            Row {
                motif_id: 0x2000000006,
                count: 3,
                percentage: 30.0,
            },
            Row {
                motif_id: 999,
                count: 2,
                percentage: 20.0,
            },
        ];
        let rows = map_motif_distribution_to_literature_basis(&distribution);
        assert_eq!(rows[0].percentage, 50.0);
        assert_eq!(rows[0].count, 5);
        assert_eq!(rows[1].percentage, 30.0);
        assert_eq!(rows[1].count, 3);
        let other = rows.last().unwrap();
        assert_eq!(other.percentage, 20.0);
        assert_eq!(other.count, 2);
    }

    #[test]
    fn formats_packed_motif_id_as_lowercase_hex() {
        assert_eq!(format_motif_hex_id(0x1000000000), "0x1000000000");
        assert_eq!(format_motif_hex_id(0x66040A0060), "0x66040a0060");
    }
}
