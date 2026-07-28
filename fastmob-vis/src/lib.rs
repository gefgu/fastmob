#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use charming::{
    Chart,
    component::{Axis, Grid, Legend, Title},
    datatype::{CompositeValue, DataPoint},
    element::{
        AxisLabel, AxisTick, AxisType, Color, LineStyle, LineStyleType, NameLocation, Padding,
        SplitLine, TextStyle, Tooltip, Trigger,
        font_settings::{FontFamily, FontWeight},
        smoothness::Smoothness,
    },
    series::Line,
};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{Value, json};

pub mod motif_svg;
pub mod motifs;

const SVG_RENDERER_UNSUPPORTED_CHART_TYPES: &[&str] = &["stvd_comparison"];

fn svg_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn value_string<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a str> {
    let mut current = value;
    for key in keys {
        current = current.get(*key)?;
    }
    current.as_str()
}

fn value_f64(value: &Value) -> Option<f64> {
    value.as_f64().filter(|v| v.is_finite())
}

fn series_array(option: &Value) -> Result<&[Value], String> {
    option
        .get("series")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or("option JSON must contain a series array".to_string())
}

fn data_point_xy(point: &Value) -> Option<(f64, f64)> {
    let values = point.as_array()?;
    if values.len() < 2 {
        return None;
    }
    Some((value_f64(&values[0])?, value_f64(&values[1])?))
}

fn collect_xy(series: &[Value]) -> Vec<(f64, f64)> {
    series
        .iter()
        .filter_map(|item| item.get("data").and_then(Value::as_array))
        .flat_map(|data| data.iter().filter_map(data_point_xy))
        .collect()
}

fn bounds(values: &[(f64, f64)]) -> (f64, f64, f64, f64) {
    let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
    for &(x, y) in values {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    if !min_x.is_finite() || min_x == max_x {
        min_x = 0.0;
        max_x = 1.0;
    }
    if !min_y.is_finite() || min_y == max_y {
        min_y = 0.0;
        max_y = 1.0;
    }
    (min_x.min(0.0), max_x, min_y.min(0.0), max_y)
}

fn render_axes(output: &mut String, width: u32, height: u32, axis_color: &str) {
    let left = 72;
    let right = width.saturating_sub(28);
    let top = 64;
    let bottom = height.saturating_sub(58);
    output.push_str(&format!(
        r#"<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="{axis_color}" stroke-width="1.5"/>"#
    ));
    output.push_str(&format!(
        r#"<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="{axis_color}" stroke-width="1.5"/>"#
    ));
}

fn render_line_like_series(
    output: &mut String,
    series: &[Value],
    width: u32,
    height: u32,
    scatter_only: bool,
) {
    let all_points = collect_xy(series);
    let (min_x, max_x, min_y, max_y) = bounds(&all_points);
    let left = 72.0;
    let right = width as f64 - 28.0;
    let top = 64.0;
    let bottom = height as f64 - 58.0;
    let sx = |x: f64| left + ((x - min_x) / (max_x - min_x)) * (right - left);
    let sy = |y: f64| bottom - ((y - min_y) / (max_y - min_y)) * (bottom - top);

    for item in series {
        let color = value_string(item, &["lineStyle", "color"])
            .or_else(|| value_string(item, &["itemStyle", "color"]))
            .unwrap_or("#111111");
        let points = item
            .get("data")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(data_point_xy)
            .map(|(x, y)| (sx(x), sy(y)))
            .collect::<Vec<_>>();
        if points.is_empty() {
            continue;
        }
        let series_type = item.get("type").and_then(Value::as_str).unwrap_or("line");
        if series_type == "line" && !scatter_only {
            let polyline = points
                .iter()
                .map(|(x, y)| format!("{x:.2},{y:.2}"))
                .collect::<Vec<_>>()
                .join(" ");
            output.push_str(&format!(
                r#"<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>"#
            ));
        }
        if series_type == "scatter" || scatter_only {
            for (x, y) in points {
                output.push_str(&format!(
                    r#"<circle cx="{x:.2}" cy="{y:.2}" r="3.5" fill="{color}"/>"#
                ));
            }
        }
    }
}

fn point_value(point: &Value) -> Option<f64> {
    point.get("value").or(Some(point)).and_then(value_f64)
}

fn render_bar_series(
    output: &mut String,
    option: &Value,
    series: &[Value],
    width: u32,
    height: u32,
) {
    let categories = option
        .get("xAxis")
        .and_then(|x| x.get("data"))
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0)
        .max(1);
    let max_value = series
        .iter()
        .filter_map(|s| s.get("data").and_then(Value::as_array))
        .flat_map(|data| data.iter())
        .filter_map(point_value)
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let left = 72.0;
    let right = width as f64 - 28.0;
    let top = 64.0;
    let bottom = height as f64 - 58.0;
    let group_width = (right - left) / categories as f64;
    let bar_width = (group_width / series.len().max(1) as f64 * 0.72).min(42.0);

    for (series_index, item) in series.iter().enumerate() {
        let color = value_string(item, &["itemStyle", "color"]).unwrap_or("#111111");
        for (index, point) in item
            .get("data")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .enumerate()
        {
            if let Some(value) = point_value(point) {
                let x = left
                    + index as f64 * group_width
                    + group_width * 0.14
                    + series_index as f64 * bar_width;
                let h = (value / max_value) * (bottom - top);
                let y = bottom - h;
                let point_color = value_string(point, &["itemStyle", "color"]).unwrap_or(color);
                output.push_str(&format!(
                    r#"<rect x="{x:.2}" y="{y:.2}" width="{bar_width:.2}" height="{h:.2}" fill="{point_color}"/>"#
                ));
            }
        }
    }
}

fn render_heatmap_series(
    output: &mut String,
    option: &Value,
    series: &[Value],
    width: u32,
    height: u32,
) {
    let x_count = option
        .get("xAxis")
        .and_then(|x| x.get("data"))
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(1)
        .max(1);
    let y_count = option
        .get("yAxis")
        .and_then(|y| y.get("data"))
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(1)
        .max(1);
    let data = series
        .first()
        .and_then(|s| s.get("data"))
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let max_value = data
        .iter()
        .filter_map(Value::as_array)
        .filter_map(|point| point.get(2).and_then(value_f64))
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let left = 72.0;
    let right = width as f64 - 96.0;
    let top = 64.0;
    let bottom = height as f64 - 58.0;
    let cell_w = (right - left) / x_count as f64;
    let cell_h = (bottom - top) / y_count as f64;
    for point in data.iter().filter_map(Value::as_array) {
        if point.len() < 3 {
            continue;
        }
        let x_index = point[0].as_u64().unwrap_or(0) as f64;
        let y_index = point[1].as_u64().unwrap_or(0) as f64;
        let value = value_f64(&point[2]).unwrap_or(0.0);
        let intensity = (value / max_value).clamp(0.0, 1.0);
        let fill = format!(
            "rgb({:.0},{:.0},{:.0})",
            245.0 - 190.0 * intensity,
            245.0 - 145.0 * intensity,
            245.0 - 80.0 * intensity
        );
        output.push_str(&format!(
            r#"<rect x="{:.2}" y="{:.2}" width="{cell_w:.2}" height="{cell_h:.2}" fill="{fill}"/>"#,
            left + x_index * cell_w,
            top + y_index * cell_h
        ));
    }
}

fn render_option_svg_value(option: Value, width: u32, height: u32) -> Result<String, String> {
    let chart_type = option
        .get("_meta")
        .and_then(|meta| meta.get("chartType"))
        .and_then(Value::as_str)
        .map(str::to_string);
    if let Some(chart_type) = chart_type.as_deref()
        && SVG_RENDERER_UNSUPPORTED_CHART_TYPES.contains(&chart_type)
    {
        return Err(format!(
            "SVG rendering is not supported for chart type {chart_type:?}"
        ));
    }
    let series = series_array(&option)?;
    if series.iter().any(|item| {
        item.get("type")
            .and_then(Value::as_str)
            .is_some_and(|kind| kind == "custom")
    }) {
        return Err("SVG rendering is not supported for custom series".to_string());
    }

    let background = option
        .get("backgroundColor")
        .and_then(Value::as_str)
        .unwrap_or("white");
    let axis_color = value_string(&option, &["textStyle", "color"]).unwrap_or("#111111");
    let title = option
        .get("title")
        .and_then(|title| {
            title
                .as_array()
                .and_then(|titles| titles.first())
                .unwrap_or(title)
                .get("text")
        })
        .and_then(Value::as_str)
        .unwrap_or("");
    let mut output = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">"#
    );
    output.push_str(&format!(
        r#"<rect width="100%" height="100%" fill="{background}"/>"#
    ));
    if !title.is_empty() {
        output.push_str(&format!(
            r#"<text x="0" y="36" fill="{axis_color}" font-size="32" font-family="sans-serif">{}</text>"#,
            svg_escape(title)
        ));
    }
    render_axes(&mut output, width, height, axis_color);

    let first_type = series
        .first()
        .and_then(|item| item.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("line");
    match first_type {
        "bar" => render_bar_series(&mut output, &option, series, width, height),
        "heatmap" => render_heatmap_series(&mut output, &option, series, width, height),
        "scatter" => render_line_like_series(&mut output, series, width, height, true),
        _ => render_line_like_series(&mut output, series, width, height, false),
    }
    output.push_str("</svg>");
    Ok(output)
}

#[pyfunction]
fn render_option_svg(option_json: String, width: u32, height: u32) -> PyResult<String> {
    let option: Value = serde_json::from_str(&option_json)
        .map_err(|error| PyValueError::new_err(format!("failed to decode option JSON: {error}")))?;
    render_option_svg_value(option, width, height).map_err(PyValueError::new_err)
}

// ---------------------------------------------------------------------------
// ECDF computation
// ---------------------------------------------------------------------------

fn validate_values(values: &[f64], name: &str) -> Result<(), String> {
    if values.is_empty() {
        return Err(format!("{name} must not be empty"));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{name} must contain only finite values"));
    }
    Ok(())
}

fn validate_cdf_cutoff(cdf_cutoff: f64) -> Result<(), String> {
    if !cdf_cutoff.is_finite() || cdf_cutoff <= 0.0 || cdf_cutoff > 1.0 {
        return Err("cdf_cutoff must be finite and in the interval (0, 1]".to_string());
    }
    Ok(())
}

pub fn ecdf_points(
    mut values: Vec<f64>,
    name: &str,
    cdf_cutoff: f64,
) -> Result<Vec<Vec<f64>>, String> {
    validate_values(&values, name)?;
    validate_cdf_cutoff(cdf_cutoff)?;
    values.sort_by(|left, right| left.total_cmp(right));

    let total = values.len() as f64;
    let mut points = Vec::new();
    let mut index = 0usize;

    while index < values.len() {
        let x = values[index];
        let mut next = index + 1;
        while next < values.len() && values[next] == x {
            next += 1;
        }
        let probability = next as f64 / total;
        if probability <= cdf_cutoff {
            points.push(vec![x, probability]);
            if probability == cdf_cutoff {
                break;
            }
        } else {
            points.push(vec![x, cdf_cutoff]);
            break;
        }
        index = next;
    }

    Ok(points)
}

#[pyfunction]
#[pyo3(signature = (values, cdf_cutoff=1.0))]
fn compute_ecdf(
    py: Python<'_>,
    values: PyReadonlyArray1<'_, f64>,
    cdf_cutoff: f64,
) -> PyResult<Vec<Vec<f64>>> {
    let vals = values.as_slice()?.to_vec();
    py.detach(move || ecdf_points(vals, "values", cdf_cutoff).map_err(PyValueError::new_err))
}

// ---------------------------------------------------------------------------
// Chart option building (charming + JSON patching)
// ---------------------------------------------------------------------------

fn legend_icon_path(dash: Option<&[u32]>, w: u32, cy: u32) -> String {
    match dash {
        None => format!("path://M0,{cy} L{w},{cy}"),
        Some(pattern) => {
            let mut segments = Vec::new();
            let mut x = 0u32;
            let mut on = true;
            let mut i = 0usize;
            while x < w {
                let seg = pattern[i % pattern.len()];
                let end = (x + seg).min(w);
                if on {
                    segments.push(format!("M{x},{cy} L{end},{cy}"));
                }
                x = end;
                on = !on;
                i += 1;
            }
            format!("path://{}", segments.join(" "))
        }
    }
}

fn points_to_dataframe(points: Vec<Vec<f64>>) -> Vec<DataPoint> {
    points
        .into_iter()
        .map(|p| DataPoint::from(CompositeValue::from(p)))
        .collect()
}

fn build_ecdf_option_value(
    series: Vec<(String, Vec<Vec<f64>>, String, Option<Vec<u32>>)>,
    title: &str,
    x_name: &str,
    x_unit: &str,
    x_label: &str,
    bg: &str,
    axis_color: &str,
    grid_color: &str,
    font_sans: &str,
    font_serif: &str,
    font_mono: &str,
) -> Result<Value, String> {
    // Capture metadata for the patch phase before series is consumed.
    let series_meta: Vec<(String, Option<Vec<u32>>)> = series
        .iter()
        .map(|(name, _, _, dash)| (name.clone(), dash.clone()))
        .collect();
    let n = series.len();

    // -----------------------------------------------------------------------
    // Build chart skeleton with charming
    // -----------------------------------------------------------------------
    let mut chart = Chart::new()
        .animation(false)
        .background_color(Color::Value(bg.to_string()))
        .title(
            Title::new().text(title).left(0i32).top(0i32).text_style(
                TextStyle::new()
                    .font_family(FontFamily::Custom(font_serif.to_string()))
                    .font_weight(FontWeight::Number(500))
                    .font_size(32.0)
                    .color(Color::Value(axis_color.to_string())),
            ),
        )
        .legend(
            Legend::new()
                .right(0i32)
                .top(12i32)
                .item_width(24.0)
                .item_height(8.0)
                .item_gap(22.0)
                .text_style(
                    TextStyle::new()
                        .font_family(FontFamily::Custom(font_sans.to_string()))
                        .font_size(15.0)
                        .color(Color::Value(axis_color.to_string())),
                ),
        )
        .grid(
            Grid::new()
                .left(76i32)
                .right(28i32)
                .top(64i32)
                .bottom(64i32)
                .contain_label(false),
        )
        .x_axis(
            Axis::new()
                .type_(AxisType::Value)
                .min(0i32)
                .split_number(5.0)
                .name(x_name)
                .name_location(NameLocation::Middle)
                .name_gap(38.0)
                .name_text_style(
                    TextStyle::new()
                        .font_family(FontFamily::Custom(font_mono.to_string()))
                        .font_size(14.0)
                        .color(Color::Value(axis_color.to_string())),
                )
                .axis_tick(
                    AxisTick::new().length(6.0).line_style(
                        LineStyle::new()
                            .color(Color::Value(axis_color.to_string()))
                            .width(1.25),
                    ),
                )
                .axis_label(
                    AxisLabel::new()
                        .font_family(FontFamily::Custom(font_mono.to_string()))
                        .font_size(16.0)
                        .color(Color::Value(axis_color.to_string())),
                )
                .split_line(SplitLine::new().show(false)),
        )
        .y_axis(
            Axis::new()
                .type_(AxisType::Value)
                .min(0i32)
                .max(1i32)
                .interval(0.2)
                .name("F(X)")
                .name_location(NameLocation::Middle)
                .name_gap(56.0)
                // name_rotation NOT set — charming uses wrong key "nameRotation";
                // we patch "nameRotate": 90 below.
                .name_text_style(
                    TextStyle::new()
                        .font_family(FontFamily::Custom(font_mono.to_string()))
                        .font_size(14.0)
                        .color(Color::Value(axis_color.to_string())),
                )
                .axis_tick(
                    AxisTick::new().length(6.0).line_style(
                        LineStyle::new()
                            .color(Color::Value(axis_color.to_string()))
                            .width(1.25),
                    ),
                )
                .axis_label(
                    AxisLabel::new()
                        .font_family(FontFamily::Custom(font_mono.to_string()))
                        .font_size(16.0)
                        .color(Color::Value(axis_color.to_string())),
                )
                .split_line(
                    SplitLine::new().line_style(
                        LineStyle::new()
                            .color(Color::Value(grid_color.to_string()))
                            .width(1.0),
                        // type patched to [2, 5] below
                    ),
                ),
        )
        .tooltip(
            Tooltip::new()
                .trigger(Trigger::Axis)
                .background_color(Color::Value(bg.to_string()))
                .border_color(Color::Value(axis_color.to_string()))
                .border_width(1.25)
                .padding(Padding::Double(10.0, 14.0)),
            // extraCssText, textStyle, axisPointer patched below
        );

    for (i, (name, points, color, dash)) in series.into_iter().enumerate() {
        let data = points_to_dataframe(points);
        let mut line_style = LineStyle::new().color(Color::Value(color)).width(2.4);
        if dash.is_some() {
            line_style = line_style.type_(LineStyleType::Dashed);
        }
        chart = chart.series(
            Line::new()
                .name(name)
                .smooth(Smoothness::Single(0.35))
                .show_symbol(false)
                .z(n as i32 - i as i32 + 1)
                .line_style(line_style)
                .data(data),
        );
    }

    // -----------------------------------------------------------------------
    // Serialize then patch everything charming can't express
    // -----------------------------------------------------------------------
    let mut value =
        serde_json::to_value(&chart).map_err(|e| format!("failed to serialize chart: {e}"))?;

    let obj = value.as_object_mut().ok_or("chart root is not an object")?;

    // Global textStyle (no field on Chart in charming)
    obj.insert(
        "textStyle".to_string(),
        json!({ "fontFamily": font_sans, "color": axis_color }),
    );

    // _meta — custom field for JS formatter context
    obj.insert(
        "_meta".to_string(),
        json!({ "xUnit": x_unit, "xLabel": x_label }),
    );

    // Unwrap grid from Vec to single object for API consistency
    if let Some(grid_val) = obj.get("grid")
        && let Some(arr) = grid_val.as_array()
        && arr.len() == 1
    {
        let single = arr[0].clone();
        obj.insert("grid".to_string(), single);
    }

    // Tooltip: extraCssText, textStyle, axisPointer
    if let Some(t) = obj.get_mut("tooltip").and_then(|t| t.as_object_mut()) {
        t.insert(
            "extraCssText".to_string(),
            json!("border-radius:0; box-shadow:none; min-width:210px;"),
        );
        t.insert("textStyle".to_string(), json!({ "color": axis_color }));
        t.insert(
            "axisPointer".to_string(),
            json!({
                "type": "line",
                "lineStyle": {
                    "color": axis_color,
                    "type": [2, 4],
                    "width": 1,
                    "opacity": 0.55,
                },
                "label": { "show": false },
            }),
        );
    }

    // xAxis: replace axisLine (charming serializes color as gradient-stop array)
    if let Some(ax) = obj.get_mut("xAxis").and_then(|a| a.as_object_mut()) {
        ax.insert(
            "axisLine".to_string(),
            json!({ "lineStyle": { "color": axis_color, "width": 1.5 } }),
        );
    }

    // yAxis: replace axisLine, fix splitLine type, add nameRotate
    if let Some(ax) = obj.get_mut("yAxis").and_then(|a| a.as_object_mut()) {
        ax.insert(
            "axisLine".to_string(),
            json!({ "lineStyle": { "color": axis_color, "width": 1.5 } }),
        );
        ax.insert("nameRotate".to_string(), json!(90));
        if let Some(ls) = ax
            .get_mut("splitLine")
            .and_then(|s| s.get_mut("lineStyle"))
            .and_then(|l| l.as_object_mut())
        {
            ls.insert("type".to_string(), json!([2, 5]));
        }
    }

    // Series: legendIcon, emphasis, cap/join, dash array replacement
    if let Some(series_arr) = obj.get_mut("series").and_then(|s| s.as_array_mut()) {
        for (i, (_, dash)) in series_meta.iter().enumerate() {
            if let Some(s) = series_arr.get_mut(i) {
                s["legendIcon"] = json!(legend_icon_path(dash.as_deref(), 24, 4));
                s["emphasis"] = json!({ "lineStyle": { "width": 2.4 } });
                if let Some(ls) = s.get_mut("lineStyle").and_then(|l| l.as_object_mut()) {
                    ls.insert("cap".to_string(), json!("round"));
                    ls.insert("join".to_string(), json!("round"));
                    if let Some(dash_arr) = dash {
                        ls.insert("type".to_string(), json!(dash_arr));
                    } else {
                        ls.remove("type");
                    }
                }
            }
        }
    }

    Ok(value)
}

#[pyfunction]
#[pyo3(signature = (series, title, x_name, x_unit, x_label, bg, axis_color, grid_color, font_sans, font_serif, font_mono))]
fn build_ecdf_option_json(
    py: Python<'_>,
    series: Vec<(String, Vec<Vec<f64>>, String, Option<Vec<u32>>)>,
    title: String,
    x_name: String,
    x_unit: String,
    x_label: String,
    bg: String,
    axis_color: String,
    grid_color: String,
    font_sans: String,
    font_serif: String,
    font_mono: String,
) -> PyResult<String> {
    py.detach(move || {
        let value = build_ecdf_option_value(
            series,
            &title,
            &x_name,
            &x_unit,
            &x_label,
            &bg,
            &axis_color,
            &grid_color,
            &font_sans,
            &font_serif,
            &font_mono,
        )
        .map_err(PyValueError::new_err)?;
        serde_json::to_string(&value)
            .map_err(|e| PyValueError::new_err(format!("failed to encode ECharts JSON: {e}")))
    })
}

// ---------------------------------------------------------------------------
// Activity chart data preparation
// ---------------------------------------------------------------------------

fn round_percent(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

fn heatmap_triplets(matrix: &[f64], rows: usize, cols: usize) -> (Vec<Value>, Option<f64>) {
    let mut data = Vec::with_capacity(matrix.len());
    let mut maximum: Option<f64> = None;
    for row in 0..rows {
        for col in 0..cols {
            let value = matrix[row * cols + col];
            if value.is_nan() {
                continue;
            }
            maximum = Some(maximum.map_or(value, |current| current.max(value)));
            data.push(json!([col, row, round_percent(value)]));
        }
    }
    (data, maximum)
}

fn option_object(option_json: &str) -> Result<Value, String> {
    let value: Value = serde_json::from_str(option_json)
        .map_err(|e| format!("failed to decode option JSON: {e}"))?;
    if !value.is_object() {
        return Err("option JSON root must be an object".to_string());
    }
    Ok(value)
}

#[pyfunction]
fn finalize_activity_bar_option_json(
    option_json: String,
    percentages: PyReadonlyArray1<'_, f64>,
    counts: Vec<i64>,
    colors: Vec<String>,
) -> PyResult<String> {
    let percentages = percentages.as_slice()?;
    if percentages.len() != counts.len() || percentages.len() != colors.len() {
        return Err(PyValueError::new_err(
            "percentages, counts, and colors must have equal length",
        ));
    }
    let mut option = option_object(&option_json).map_err(PyValueError::new_err)?;
    let data: Vec<Value> = percentages
        .iter()
        .zip(counts)
        .zip(colors)
        .map(|((&value, count), color)| {
            json!({
                "value": round_percent(value),
                "itemStyle": {"color": color},
                "count": count,
            })
        })
        .collect();
    option["series"][0]["data"] = Value::Array(data);
    serde_json::to_string(&option)
        .map_err(|e| PyValueError::new_err(format!("failed to encode activity option JSON: {e}")))
}

#[pyfunction]
fn finalize_activity_comparison_option_json(
    option_json: String,
    percentages: PyReadonlyArray2<'_, f64>,
    counts: Vec<Vec<Option<i64>>>,
    colors: Vec<String>,
    opacities: Vec<f64>,
) -> PyResult<String> {
    let percentages = percentages.as_array();
    let rows = percentages.nrows();
    let cols = percentages.ncols();
    if counts.len() != rows
        || counts.iter().any(|row| row.len() != cols)
        || colors.len() != cols
        || opacities.len() != rows
    {
        return Err(PyValueError::new_err(
            "comparison percentages, counts, colors, and opacities have incompatible dimensions",
        ));
    }

    let mut option = option_object(&option_json).map_err(PyValueError::new_err)?;
    let series = option["series"]
        .as_array_mut()
        .ok_or_else(|| PyValueError::new_err("option JSON must contain a series array"))?;
    if series.len() != rows {
        return Err(PyValueError::new_err(
            "comparison percentages must have one row per series",
        ));
    }

    for row in 0..rows {
        let data = (0..cols)
            .map(|col| {
                let mut item = json!({
                    "value": round_percent(percentages[[row, col]]),
                    "itemStyle": {
                        "color": colors[col],
                        "opacity": opacities[row],
                    },
                });
                if let Some(count) = counts[row][col] {
                    item["count"] = json!(count);
                }
                item
            })
            .collect();
        series[row]["data"] = Value::Array(data);
    }

    serde_json::to_string(&option).map_err(|e| {
        PyValueError::new_err(format!(
            "failed to encode activity comparison option JSON: {e}"
        ))
    })
}

#[pyfunction]
#[pyo3(signature = (option_json, matrix, update_visual_max=false))]
fn finalize_activity_heatmap_option_json(
    option_json: String,
    matrix: PyReadonlyArray2<'_, f64>,
    update_visual_max: bool,
) -> PyResult<String> {
    let matrix = matrix.as_array();
    let owned;
    let values = if let Some(slice) = matrix.as_slice() {
        slice
    } else {
        owned = matrix.iter().copied().collect::<Vec<_>>();
        &owned
    };
    let (data, maximum) = heatmap_triplets(values, matrix.nrows(), matrix.ncols());
    let mut option = option_object(&option_json).map_err(PyValueError::new_err)?;
    option["series"][0]["data"] = Value::Array(data);
    if update_visual_max {
        option["visualMap"]["max"] = json!(maximum.unwrap_or(100.0).max(1.0));
    }
    serde_json::to_string(&option)
        .map_err(|e| PyValueError::new_err(format!("failed to encode activity option JSON: {e}")))
}

#[pyfunction]
fn activity_time_bin_labels(n_bins: usize) -> PyResult<Vec<String>> {
    if n_bins == 0 || 1440 % n_bins != 0 {
        return Err(PyValueError::new_err(
            "n_bins must be a positive divisor of 1440",
        ));
    }
    let minutes_per_bin = 1440 / n_bins;
    Ok((0..n_bins)
        .map(|idx| {
            let minute = idx * minutes_per_bin;
            format!("{:02}:{:02}", minute / 60, minute % 60)
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Python module
// ---------------------------------------------------------------------------

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_ecdf, m)?)?;
    m.add_function(wrap_pyfunction!(build_ecdf_option_json, m)?)?;
    m.add_function(wrap_pyfunction!(finalize_activity_bar_option_json, m)?)?;
    m.add_function(wrap_pyfunction!(
        finalize_activity_comparison_option_json,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(finalize_activity_heatmap_option_json, m)?)?;
    m.add_function(wrap_pyfunction!(activity_time_bin_labels, m)?)?;
    m.add_function(wrap_pyfunction!(render_option_svg, m)?)?;
    Ok(())
}

#[cfg(test)]
mod activity_tests {
    use super::*;

    #[test]
    fn heatmap_preparation_rounds_and_omits_nan() {
        let (data, maximum) = heatmap_triplets(&[33.333333, f64::NAN, 75.0, 0.0], 2, 2);
        assert_eq!(
            data,
            vec![
                json!([0, 0, 33.33]),
                json!([0, 1, 75.0]),
                json!([1, 1, 0.0])
            ]
        );
        assert_eq!(maximum, Some(75.0));
    }

    #[test]
    fn empty_heatmap_has_no_maximum() {
        let (data, maximum) = heatmap_triplets(&[f64::NAN], 1, 1);
        assert!(data.is_empty());
        assert_eq!(maximum, None);
    }

    #[test]
    fn renders_option_json_to_svg() {
        let svg = render_option_svg_value(
            json!({
                "animation": false,
                "backgroundColor": "white",
                "xAxis": {"type": "value"},
                "yAxis": {"type": "value"},
                "series": [{"type": "line", "data": [[0, 0], [1, 1]]}],
            }),
            320,
            240,
        )
        .unwrap();

        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn rejects_unsupported_svg_chart_type() {
        let error = render_option_svg_value(
            json!({"_meta": {"chartType": "stvd_comparison"}, "series": []}),
            320,
            240,
        )
        .unwrap_err();

        assert!(error.contains("not supported"));
    }
}
