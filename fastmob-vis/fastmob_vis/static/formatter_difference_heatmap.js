  function formatDifference(value, suffix) {
    const number = Number(value);
    if (!Number.isFinite(number)) return '';
    const sign = number > 0 ? '+' : '';
    return sign + number.toFixed(2) + (suffix || '');
  }
  function heatmapAxisData(axis) {
    if (Array.isArray(axis)) axis = axis[0] || {};
    return axis.data || [];
  }
  // Labels are black except for cells in the lowest 30% of the value range, whose
  // near-black fill needs white text to stay readable.
  function labelColorForValue(value, baseColor) {
    const visualMap = Array.isArray(option.visualMap) ? option.visualMap[0] : option.visualMap;
    if (visualMap) {
      const min = Number(visualMap.min);
      const max = Number(visualMap.max);
      if (max > min) {
        const t = (Number(value) - min) / (max - min);
        if (t <= 0.3) return '#ffffff';
      }
    }
    return baseColor;
  }
  option.tooltip.formatter = function(params) {
    params = Array.isArray(params) ? params[0] : params;
    if (!params || !params.value) return '';
    const value = params.value;
    const xLabels = heatmapAxisData(option.xAxis);
    const yLabels = heatmapAxisData(option.yAxis);
    const xLabel = xLabels[value[0]] || String(value[0]);
    const yLabel = yLabels[value[1]] || String(value[1]);
    const labels = meta.differenceLabels || ['first', 'second'];
    const direction = labels[1] + ' - ' + labels[0];
    const difference = formatDifference(value[2], ' pp');
    if (meta.chartType === 'transition_difference') {
      return yLabel + ' -> ' + xLabel + '<br/>' + direction + ': ' + difference;
    }
    return yLabel + '<br/>' + xLabel + '<br/>' + direction + ': ' + difference;
  };
  option.series.forEach(function(series) {
    if (series.label) {
      const baseColor = typeof series.label.color === 'string' ? series.label.color : '#14110d';
      const labelFormatter = function(params) {
        return formatDifference(params.value[2]);
      };
      const labelColor = function(params) {
        return labelColorForValue(params.value[2], baseColor);
      };
      series.label.formatter = labelFormatter;
      series.label.color = labelColor;
      // Keep the label visible and readable while hovering; without an explicit
      // emphasis label ECharts drops the cell text on hover.
      series.emphasis = series.emphasis || {};
      series.emphasis.label = Object.assign({}, series.emphasis.label, {
        show: true,
        formatter: labelFormatter,
        color: labelColor,
      });
    }
  });
