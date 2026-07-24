  const xUnit = meta.xUnit || '';
  option.xAxis.axisLabel.formatter = function(v) {
    return Number.isInteger(v) ? String(v) : parseFloat(v.toFixed(2)).toString();
  };
  option.yAxis.axisLabel.formatter = function(v) {
    if (v === 0 || v === 1) return String(v);
    return parseFloat(v.toFixed(1)).toString();
  };
  function ecdfAt(data, x) {
    var lo = 0, hi = data.length - 1, last = 0;
    while (lo <= hi) {
      var mid = (lo + hi) >> 1;
      if (data[mid][0] <= x) { last = data[mid][1]; lo = mid + 1; }
      else { hi = mid - 1; }
    }
    return last;
  }
  option.tooltip.formatter = function(params) {
    if (!params.length) return '';
    const x = Number(params[0].axisValue);
    const xStr = Number.isInteger(x) ? String(x) : x.toFixed(2);
    const xHeader = xUnit ? 'X = ' + xStr + ' ' + xUnit.toUpperCase() : 'X = ' + xStr;
    let out = '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:12px;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:6px;">' + xHeader + '</div>';
    option.series.forEach(function(s) {
      const fy = ecdfAt(s.data, x);
      const ls = s.lineStyle || {};
      const isDashed = Array.isArray(ls.type);
      const borderStyle = isDashed ? 'dashed' : 'solid';
      const color = ls.color || '#000';
      out += '<div style="display:flex;align-items:center;gap:10px;padding:2px 0;">';
      out += '<span style="display:inline-block;width:20px;border-top:2.5px ' + borderStyle + ' ' + color + ';flex-shrink:0;"></span>';
      out += '<span style="flex:1;font-family:\'IBM Plex Sans\',sans-serif;font-size:14px;">' + s.name + '</span>';
      out += '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:14px;font-weight:500;font-variant-numeric:tabular-nums;">' + fy.toFixed(3) + '</span>';
      out += '</div>';
    });
    return out;
  };
