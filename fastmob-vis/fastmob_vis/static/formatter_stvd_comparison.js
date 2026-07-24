  const stvdLayers = meta.layers || {};
  const stvdZoomMapping = meta.zoomToResolution || [];
  const stvdColors = meta.colors || [];
  let stvdResolution = meta.initialResolution;
  let stvdFeatures = (stvdLayers[String(stvdResolution)] || {}).features || [];

  Object.keys(stvdLayers).forEach(function(resolution) {
    echarts.registerMap('fastmob-stvd-res-' + resolution, stvdLayers[resolution]);
  });

  function stvdPolygonShape(rings, api) {
    return {
      type: 'polygon',
      shape: {
        points: rings[0].map(function(coordinate) {
          return api.coord(coordinate);
        })
      }
    };
  }

  option.series[0].renderItem = function(params, api) {
    const feature = stvdFeatures[api.value(0)];
    if (!feature || !feature.geometry) return null;
    const geometry = feature.geometry;
    const polygons = geometry.type === 'Polygon' ? [geometry.coordinates] : geometry.coordinates;
    const style = feature.properties._fastmobVis;
    return {
      type: 'group',
      children: polygons.map(function(polygon) {
        const shape = stvdPolygonShape(polygon, api);
        shape.style = {
          fill: style.color,
          stroke: '#14110d',
          lineWidth: stvdResolution <= 5 ? 0.9 : 0.45,
          opacity: 0.9
        };
        shape.emphasis = {
          style: {
            fill: style.color,
            stroke: '#14110d',
            lineWidth: 1.5,
            opacity: 1
          }
        };
        return shape;
      })
    };
  };

  function stvdEscape(value) {
    return String(value)
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#39;');
  }

  option.tooltip.formatter = function(params) {
    const feature = stvdFeatures[params.value[0]];
    if (!feature) return '';
    const value = feature.properties._fastmobVis;
    const volumeLabels = [
      '< -' + meta.volumeThreshold + '%',
      '\u00b1' + meta.volumeThreshold + '%',
      '> ' + meta.volumeThreshold + '%'
    ];
    const peakLabels = ['0\u20132 h', '3\u20135 h', '6\u201312 h'];
    return '<strong>' + stvdEscape(value.area) + '</strong><br/>'
      + 'H3 resolution: ' + stvdResolution + '<br/>'
      + 'Volume difference: ' + value.volumeDiff.toFixed(2) + '% ('
      + volumeLabels[value.volumeBin] + ')<br/>'
      + 'Peak shift: ' + value.peakShift.toFixed(2) + ' h ('
      + peakLabels[value.peakBin] + ')';
  };

  function stvdResolutionForZoom(zoom) {
    let resolution = stvdZoomMapping[0][1];
    stvdZoomMapping.forEach(function(entry) {
      if (zoom >= entry[0]) resolution = entry[1];
    });
    return resolution;
  }

  function stvdSeriesData(features) {
    return features.map(function(_feature, index) { return [index]; });
  }

  function stvdLegend(container) {
    const legend = document.createElement('div');
    legend.className = 'fastmob-vis-stvd-legend';
    const peakLabels = ['0\u20132 h', '3\u20135 h', '6\u201312 h'];
    const volumeLabels = [
      '< -' + meta.volumeThreshold + '%',
      '\u00b1' + meta.volumeThreshold + '%',
      '> ' + meta.volumeThreshold + '%'
    ];

    // Column 1 holds the rotated y-axis title, column 2 the y tick labels, and
    // columns 3-5 the 3x3 color grid. Every element is positioned explicitly so
    // auto-placement can never shuffle labels onto the color cells.
    const yTitle = document.createElement('div');
    yTitle.className = 'fastmob-vis-stvd-legend__y-title';
    yTitle.textContent = 'Peak shift \u2191';
    legend.appendChild(yTitle);

    for (let row = 2; row >= 0; row -= 1) {
      const gridRow = 3 - row;
      const yLabel = document.createElement('div');
      yLabel.className = 'fastmob-vis-stvd-legend__y';
      yLabel.textContent = peakLabels[row];
      yLabel.style.gridColumn = '2';
      yLabel.style.gridRow = String(gridRow);
      legend.appendChild(yLabel);
      for (let column = 0; column < 3; column += 1) {
        const cell = document.createElement('div');
        cell.className = 'fastmob-vis-stvd-legend__cell';
        cell.style.background = stvdColors[row][column];
        cell.style.gridColumn = String(3 + column);
        cell.style.gridRow = String(gridRow);
        legend.appendChild(cell);
      }
    }

    volumeLabels.forEach(function(label, column) {
      const xLabel = document.createElement('div');
      xLabel.className = 'fastmob-vis-stvd-legend__x';
      xLabel.textContent = label;
      xLabel.style.gridColumn = String(3 + column);
      xLabel.style.gridRow = '4';
      legend.appendChild(xLabel);
    });

    const xTitle = document.createElement('div');
    xTitle.className = 'fastmob-vis-stvd-legend__x-title';
    xTitle.textContent = 'Volume difference \u2192';
    legend.appendChild(xTitle);

    container.appendChild(legend);
  }

  var fastmobVisInitialize = function(chart, container) {
    stvdLegend(container);
    const leafletMap = chart.getModel().getComponent('leaflet').getLeaflet();
    leafletMap.on('zoomend', function() {
      const targetResolution = stvdResolutionForZoom(leafletMap.getZoom());
      if (targetResolution === stvdResolution) return;
      stvdResolution = targetResolution;
      stvdFeatures = stvdLayers[String(stvdResolution)].features;
      chart.setOption({
        series: [{
          data: stvdSeriesData(stvdFeatures)
        }]
      });
    });
  };
