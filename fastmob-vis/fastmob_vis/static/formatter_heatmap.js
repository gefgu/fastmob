  function heatmapAxisData(axis) {
    if (Array.isArray(axis)) axis = axis[0] || {};
    return axis.data || [];
  }
  option.tooltip.formatter = function(params) {
    params = Array.isArray(params) ? params[0] : params;
    if (!params || !params.value) return '';
    const value = params.value;
    const xLabels = heatmapAxisData(option.xAxis);
    const yLabels = heatmapAxisData(option.yAxis);
    const xLabel = xLabels[value[0]] || String(value[0]);
    const yLabel = yLabels[value[1]] || String(value[1]);
    const percent = formatPercent(value[2]);
    if (meta.chartType === 'transition') {
      return yLabel + ' -> ' + xLabel + '<br/>Percentage: ' + percent;
    }
    return yLabel + '<br/>' + xLabel + '<br/>Percentage: ' + percent;
  };
  option.series.forEach(function(series) {
    if (series.label) {
      series.label.formatter = function(params) {
        return formatPercent(params.value[2]);
      };
    }
  });
