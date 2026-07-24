  option.tooltip.formatter = function(params) {
    if (!Array.isArray(params) || !params.length) return '';
    let out = params[0].name;
    params.forEach(function(param) {
      const data = param.data || {};
      out += '<br/>' + param.marker + param.seriesName + ': ' + formatPercent(data.value);
      if (data.count !== undefined && data.count !== null) {
        out += ' (' + data.count + ' visits)';
      }
    });
    return out;
  };
