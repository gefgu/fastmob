  option.tooltip.formatter = function(params) {
    params = Array.isArray(params) ? params[0] : params;
    if (!params) return '';
    return params.name + '<br/>' + params.seriesName + ': ' + formatPercent(params.value);
  };
  option.series.forEach(function(series) {
    if (series.label) {
      series.label.formatter = function(params) {
        return formatPercent(params.value);
      };
    }
  });
