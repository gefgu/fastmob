use fastmob_vis::ecdf_points;

#[test]
fn ecdf_sorts_values() {
    let points = ecdf_points(vec![2.0, 1.0, 3.0], "values", 1.0).unwrap();
    assert_eq!(
        points,
        vec![
            vec![1.0, 1.0 / 3.0],
            vec![2.0, 2.0 / 3.0],
            vec![3.0, 1.0]
        ]
    );
}

#[test]
fn ecdf_compresses_duplicates() {
    let points = ecdf_points(vec![2.0, 1.0, 1.0, 2.0], "values", 1.0).unwrap();
    assert_eq!(points, vec![vec![1.0, 0.5], vec![2.0, 1.0]]);
}

#[test]
fn ecdf_applies_cdf_cutoff() {
    let points = ecdf_points(vec![4.0, 1.0, 3.0, 2.0], "values", 0.75).unwrap();
    assert_eq!(
        points,
        vec![vec![1.0, 0.25], vec![2.0, 0.5], vec![3.0, 0.75]]
    );
}

#[test]
fn ecdf_caps_first_point_when_it_crosses_cutoff() {
    let points = ecdf_points(vec![1.0], "values", 0.98).unwrap();
    assert_eq!(points, vec![vec![1.0, 0.98]]);
}

#[test]
fn ecdf_rejects_empty_values() {
    let error = ecdf_points(Vec::new(), "values", 1.0).unwrap_err();
    assert!(error.contains("must not be empty"));
}

#[test]
fn ecdf_rejects_non_finite_values() {
    let error = ecdf_points(vec![1.0, f64::NAN], "values", 1.0).unwrap_err();
    assert!(error.contains("finite"));
}

#[test]
fn ecdf_rejects_invalid_cdf_cutoff() {
    let error = ecdf_points(vec![1.0], "values", 0.0).unwrap_err();
    assert!(error.contains("cdf_cutoff"));
}
