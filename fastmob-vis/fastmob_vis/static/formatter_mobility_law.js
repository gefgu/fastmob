  function formatLawNumber(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return String(value);
    const magnitude = Math.abs(number);
    if ((magnitude > 0 && magnitude < 0.001) || magnitude >= 10000) {
      return number.toExponential(3);
    }
    return parseFloat(number.toPrecision(4)).toString();
  }
  function fitParameterText(parameters) {
    if (!parameters) return '';
    return Object.keys(parameters).map(function(key) {
      return key + ' = ' + formatLawNumber(parameters[key]);
    }).join(', ');
  }
  option.xAxis.axisLabel.formatter = formatLawNumber;
  option.yAxis.axisLabel.formatter = formatLawNumber;
  option.tooltip.formatter = function(params) {
    const value = params.value || [];
    let out = '<div style="font-family:\'IBM Plex Sans\',sans-serif;font-size:14px;">' + params.seriesName + '</div>';
    out += '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:13px;margin-top:4px;">';
    out += 'x = ' + formatLawNumber(value[0]) + '<br/>y = ' + formatLawNumber(value[1]) + '</div>';
    const series = option.series[params.seriesIndex] || {};
    const parameterText = fitParameterText(series.fitParameters);
    if (parameterText) {
      out += '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:12px;margin-top:6px;">' + parameterText + '</div>';
    }
    return out;
  };
