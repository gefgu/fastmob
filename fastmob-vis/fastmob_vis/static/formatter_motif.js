  const motifLabelKeys = meta.motifLabelKeys || {};
  option.xAxis.axisLabel.formatter = function(value) {
    const styleKey = motifLabelKeys[value];
    return styleKey ? '{' + styleKey + '| }' : value;
  };
  option.tooltip.formatter = function(params) {
    const value = params.value || [];
    return 'Literature motif: ' + value[2] + '<br/>'
      + 'Packed motif ID: ' + value[3] + '<br/>'
      + 'Hex ID: ' + value[4] + '<br/>'
      + params.seriesName + ': ' + formatPercent(value[1]) + '<br/>'
      + 'Count: ' + value[5];
  };
