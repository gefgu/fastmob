(function (global, factory) {
	typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports, require('echarts/lib/echarts'), require('leaflet/src/Leaflet')) :
	typeof define === 'function' && define.amd ? define(['exports', 'echarts/lib/echarts', 'leaflet/src/Leaflet'], factory) :
	(global = typeof globalThis !== 'undefined' ? globalThis : global || self, factory(global.leaflet = {}, global.echarts, global.L));
})(this, (function (exports, echarts, L) { 'use strict';

	// 这段代码来自于 https://github.com/Leaflet/Leaflet.TileLayer.NoGap/blob/master/L.TileLayer.NoGap.js
	// 如果没有这段，则显示的地图的瓦片之间有空白线

	// @class TileLayer
	L.TileLayer.mergeOptions({
		// @option keepBuffer
		// The amount of tiles outside the visible map area to be kept in the stitched
		// `TileLayer`.

		// @option dumpToCanvas: Boolean = true
		// Whether to dump loaded tiles to a `<canvas>` to prevent some rendering
		// artifacts. (Disabled by default in IE)
		dumpToCanvas: L.Browser.canvas && !L.Browser.ie
	});

	L.TileLayer.include({
		_onUpdateLevel: function _onUpdateLevel(z, zoom) {
			if (this.options.dumpToCanvas) {
				this._levels[z].canvas.style.zIndex = this.options.maxZoom - Math.abs(zoom - z);
			}
		},

		_onRemoveLevel: function _onRemoveLevel(z) {
			if (this.options.dumpToCanvas) {
				L.DomUtil.remove(this._levels[z].canvas);
			}
		},

		_onCreateLevel: function _onCreateLevel(level) {
			if (this.options.dumpToCanvas) {
				level.canvas = L.DomUtil.create("canvas", "leaflet-tile-container leaflet-zoom-animated", this._container);
				level.ctx = level.canvas.getContext("2d");
				this._resetCanvasSize(level);
			}
		},

		_removeTile: function _removeTile(key) {
			if (this.options.dumpToCanvas) {
				var tile = this._tiles[key];
				var level = this._levels[tile.coords.z];
				var tileSize = this.getTileSize();

				if (level) {
					// Where in the canvas should this tile go?
					var offset = L.point(tile.coords.x, tile.coords.y).subtract(level.canvasRange.min).scaleBy(this.getTileSize());

					level.ctx.clearRect(offset.x, offset.y, tileSize.x, tileSize.y);
				}
			}

			L.GridLayer.prototype._removeTile.call(this, key);
		},

		_resetCanvasSize: function _resetCanvasSize(level) {
			var buff = this.options.keepBuffer,
			    pixelBounds = this._getTiledPixelBounds(this._map.getCenter()),
			    tileRange = this._pxBoundsToTileRange(pixelBounds),
			    tileSize = this.getTileSize();

			tileRange.min = tileRange.min.subtract([buff, buff]); // This adds the no-prune buffer
			tileRange.max = tileRange.max.add([buff + 1, buff + 1]);

			var pixelRange = L.bounds(tileRange.min.scaleBy(tileSize), tileRange.max.add([1, 1]).scaleBy(tileSize) // This prevents an off-by-one when checking if tiles are inside
			),
			    mustRepositionCanvas = false,
			    neededSize = pixelRange.max.subtract(pixelRange.min);

			// Resize the canvas, if needed, and only to make it bigger.
			if (neededSize.x > level.canvas.width || neededSize.y > level.canvas.height) {
				// Resizing canvases erases the currently drawn content, I'm afraid.
				// To keep it, dump the pixels to another canvas, then display it on
				// top. This could be done with getImageData/putImageData, but that
				// would break for tainted canvases (in non-CORS tilesets)
				var oldSize = { x: level.canvas.width, y: level.canvas.height };
				// console.info('Resizing canvas from ', oldSize, 'to ', neededSize);

				var tmpCanvas = L.DomUtil.create("canvas");
				tmpCanvas.style.width = (tmpCanvas.width = oldSize.x) + "px";
				tmpCanvas.style.height = (tmpCanvas.height = oldSize.y) + "px";
				tmpCanvas.getContext("2d").drawImage(level.canvas, 0, 0);
				// var data = level.ctx.getImageData(0, 0, oldSize.x, oldSize.y);

				level.canvas.style.width = (level.canvas.width = neededSize.x) + "px";
				level.canvas.style.height = (level.canvas.height = neededSize.y) + "px";
				level.ctx.drawImage(tmpCanvas, 0, 0);
				// level.ctx.putImageData(data, 0, 0, 0, 0, oldSize.x, oldSize.y);
			}

			// Translate the canvas contents if it's moved around
			if (level.canvasRange) {
				var offset = level.canvasRange.min.subtract(tileRange.min).scaleBy(this.getTileSize());

				// 			console.info('Offsetting by ', offset);

				if (!L.Browser.safari) {
					// By default, canvases copy things "on top of" existing pixels, but we want
					// this to *replace* the existing pixels when doing a drawImage() call.
					// This will also clear the sides, so no clearRect() calls are needed to make room
					// for the new tiles.
					level.ctx.globalCompositeOperation = "copy";
					level.ctx.drawImage(level.canvas, offset.x, offset.y);
					level.ctx.globalCompositeOperation = "source-over";
				} else {
					// Safari clears the canvas when copying from itself :-(
					if (!this._tmpCanvas) {
						var t = this._tmpCanvas = L.DomUtil.create("canvas");
						t.width = level.canvas.width;
						t.height = level.canvas.height;
						this._tmpContext = t.getContext("2d");
					}
					this._tmpContext.clearRect(0, 0, level.canvas.width, level.canvas.height);
					this._tmpContext.drawImage(level.canvas, 0, 0);
					level.ctx.clearRect(0, 0, level.canvas.width, level.canvas.height);
					level.ctx.drawImage(this._tmpCanvas, offset.x, offset.y);
				}

				mustRepositionCanvas = true; // Wait until new props are set
			}

			level.canvasRange = tileRange;
			level.canvasPxRange = pixelRange;
			level.canvasOrigin = pixelRange.min;

			// console.log('Canvas tile range: ', level, tileRange.min, tileRange.max );
			// console.log('Canvas pixel range: ', pixelRange.min, pixelRange.max );
			// console.log('Level origin: ', level.origin );

			if (mustRepositionCanvas) {
				this._setCanvasZoomTransform(level, this._map.getCenter(), this._map.getZoom());
			}
		},

		/// set transform/position of canvas, in addition to the transform/position of the individual tile container
		_setZoomTransform: function _setZoomTransform(level, center, zoom) {
			L.GridLayer.prototype._setZoomTransform.call(this, level, center, zoom);
			if (this.options.dumpToCanvas) {
				this._setCanvasZoomTransform(level, center, zoom);
			}
		},

		// This will get called twice:
		// * From _setZoomTransform
		// * When the canvas has shifted due to a new tile being loaded
		_setCanvasZoomTransform: function _setCanvasZoomTransform(level, center, zoom) {
			// console.log('_setCanvasZoomTransform', level, center, zoom);
			if (!level.canvasOrigin) {
				return;
			}
			var scale = this._map.getZoomScale(zoom, level.zoom),
			    translate = level.canvasOrigin.multiplyBy(scale).subtract(this._map._getNewPixelOrigin(center, zoom)).round();

			if (L.Browser.any3d) {
				L.DomUtil.setTransform(level.canvas, translate, scale);
			} else {
				L.DomUtil.setPosition(level.canvas, translate);
			}
		},

		_onOpaqueTile: function _onOpaqueTile(tile) {
			if (!this.options.dumpToCanvas) {
				return;
			}

			// Guard against an NS_ERROR_NOT_AVAILABLE (or similar) exception
			// when a non-image-tile has been loaded (e.g. a WMS error).
			// Checking for tile.el.complete is not enough, as it has been
			// already marked as loaded and ready somehow.
			try {
				this.dumpPixels(tile.coords, tile.el);
			} catch (ex) {
				return this.fire("tileerror", {
					error: "Could not copy tile pixels: " + ex,
					tile: tile,
					coods: tile.coords
				});
			}

			// If dumping the pixels was successful, then hide the tile.
			// Do not remove the tile itself, as it is needed to check if the whole
			// level (and its canvas) should be removed (via level.el.children.length)
			tile.el.style.display = "none";
		},

		// @section Extension methods
		// @uninheritable

		// @method dumpPixels(coords: Object, imageSource: CanvasImageSource): this
		// Dumps pixels from the given `CanvasImageSource` into the layer, into
		// the space for the tile represented by the `coords` tile coordinates (an object
		// like `{x: Number, y: Number, z: Number}`; the image source must have the
		// same size as the `tileSize` option for the layer. Has no effect if `dumpToCanvas`
		// is `false`.
		dumpPixels: function dumpPixels(coords, imageSource) {
			var level = this._levels[coords.z],
			    tileSize = this.getTileSize();

			if (!level.canvasRange || !this.options.dumpToCanvas) {
				return;
			}

			// Check if the tile is inside the currently visible map bounds
			// There is a possible race condition when tiles are loaded after they
			// have been panned outside of the map.
			if (!level.canvasRange.contains(coords)) {
				this._resetCanvasSize(level);
			}

			// Where in the canvas should this tile go?
			var offset = L.point(coords.x, coords.y).subtract(level.canvasRange.min).scaleBy(this.getTileSize());

			level.ctx.drawImage(imageSource, offset.x, offset.y, tileSize.x, tileSize.y);

			// TODO: Clear the pixels of other levels' canvases where they overlap
			// this newly dumped tile.
			return this;
		}
	});

	var classCallCheck = function (instance, Constructor) {
	  if (!(instance instanceof Constructor)) {
	    throw new TypeError("Cannot call a class as a function");
	  }
	};

	var createClass = function () {
	  function defineProperties(target, props) {
	    for (var i = 0; i < props.length; i++) {
	      var descriptor = props[i];
	      descriptor.enumerable = descriptor.enumerable || false;
	      descriptor.configurable = true;
	      if ("value" in descriptor) descriptor.writable = true;
	      Object.defineProperty(target, descriptor.key, descriptor);
	    }
	  }

	  return function (Constructor, protoProps, staticProps) {
	    if (protoProps) defineProperties(Constructor.prototype, protoProps);
	    if (staticProps) defineProperties(Constructor, staticProps);
	    return Constructor;
	  };
	}();

	var get = function get(object, property, receiver) {
	  if (object === null) object = Function.prototype;
	  var desc = Object.getOwnPropertyDescriptor(object, property);

	  if (desc === undefined) {
	    var parent = Object.getPrototypeOf(object);

	    if (parent === null) {
	      return undefined;
	    } else {
	      return get(parent, property, receiver);
	    }
	  } else if ("value" in desc) {
	    return desc.value;
	  } else {
	    var getter = desc.get;

	    if (getter === undefined) {
	      return undefined;
	    }

	    return getter.call(receiver);
	  }
	};

	var inherits = function (subClass, superClass) {
	  if (typeof superClass !== "function" && superClass !== null) {
	    throw new TypeError("Super expression must either be null or a function, not " + typeof superClass);
	  }

	  subClass.prototype = Object.create(superClass && superClass.prototype, {
	    constructor: {
	      value: subClass,
	      enumerable: false,
	      writable: true,
	      configurable: true
	    }
	  });
	  if (superClass) Object.setPrototypeOf ? Object.setPrototypeOf(subClass, superClass) : subClass.__proto__ = superClass;
	};

	var possibleConstructorReturn = function (self, call) {
	  if (!self) {
	    throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
	  }

	  return call && (typeof call === "object" || typeof call === "function") ? call : self;
	};

	// 来源这里： https://gitee.com/qwswqwe/leaflet-user-define-theme
	// 对应的文章在这里：https://blog.csdn.net/qq_58771242/article/details/132981229
	// 用于修改图层的颜色，如果在options中有color选项，则会使用该颜色来对图片的颜色做反转
	// 有了这个思路之后，如果以后需要对瓦片做什么特殊处理，比如叠加显示一个特定的文本，也是可以的


	var DesignTileLayer = function (_L$TileLayer) {
	  inherits(DesignTileLayer, _L$TileLayer);

	  function DesignTileLayer() {
	    classCallCheck(this, DesignTileLayer);
	    return possibleConstructorReturn(this, (DesignTileLayer.__proto__ || Object.getPrototypeOf(DesignTileLayer)).apply(this, arguments));
	  }

	  createClass(DesignTileLayer, [{
	    key: "createTile",
	    value: function createTile(coords, done) {
	      var _this2 = this;

	      var color = this.options.color;

	      if (!color) {
	        return get(DesignTileLayer.prototype.__proto__ || Object.getPrototypeOf(DesignTileLayer.prototype), "createTile", this).call(this, coords, done);
	      }
	      /**获取外部接受的颜色*/
	      console.warn("createTile", this.options);

	      console.warn("createTile color", color);

	      // 创建一个用于绘图的 <canvas> 元素
	      var tile = L.DomUtil.create("canvas", "leaflet-tile");

	      // 根据选项设置瓦片的宽度和高度
	      var size = this.getTileSize();
	      tile.width = size.x;
	      tile.height = size.y;

	      // 获得一个 canvas 上下文，并使用 coords.x、coords.y 和 coords.z 在上面画东西
	      var ctx = tile.getContext("2d");

	      // 使用传入的 URL 模板替换变量
	      var url = this.getTileUrl(coords);
	      // 创建一个图像对象来加载瓦片
	      var img = new Image();
	      // 解决跨域的问题，https://blog.csdn.net/leftfist/article/details/106947101
	      img.setAttribute("crossOrigin", "");
	      // 以下是TileLayer的createTile方法内对于跨域的处理，可以考虑参考下
	      // https://github.com/Leaflet/Leaflet/blob/21d27856f2e314d267ea6120363eb60660bae123/src/layer/tile/TileLayer.js#L145
	      //   if (this.options.crossOrigin || this.options.crossOrigin === "") {
	      //     tile.crossOrigin =
	      //       this.options.crossOrigin === true ? "" : this.options.crossOrigin;
	      //   }

	      //   // for this new option we follow the documented behavior
	      //   // more closely by only setting the property when string
	      //   if (typeof this.options.referrerPolicy === "string") {
	      //     tile.referrerPolicy = this.options.referrerPolicy;
	      //   }
	      img.src = url; // 替换为你的图片路径

	      // 当图片加载完成后，绘制到 Canvas 上
	      img.onload = function () {
	        // 绘制图片到 Canvas 上
	        console.warn("img onLoad", tile.width, tile.height);
	        ctx.drawImage(img, 0, 0, tile.width, tile.height);
	        // 获取图像的像素数据
	        var imageData = ctx.getImageData(0, 0, tile.width, tile.height);
	        if (color) {
	          // 获取原来的图片的像素颜色
	          var pixels = imageData.data;
	          for (var i = 0; i < pixels.length; i += 4) {
	            var r = pixels[i],
	                g = pixels[i + 1],
	                b = pixels[i + 2],
	                a = pixels[i + 3];
	            //计算灰度
	            var grayVal = (r + g + b) / 3;
	            //灰度反转--会使图片整体变成灰色--方便上色
	            grayVal = 255 - grayVal;
	            //将灰度替换掉原始的颜色
	            pixels[i] = grayVal + color.r;
	            pixels[i + 1] = grayVal + color.g;
	            pixels[i + 2] = grayVal + color.b;
	            //设置一个前景透明度，以便和背景混合
	            if (color.a) {
	              pixels[i + 3] = a * color.a;
	            }
	          }
	        }

	        // 将修改后的像素数据放回 Canvas
	        ctx.putImageData(imageData, 0, 0);
	        _this2._tileOnLoad(done, img);
	      };
	      // 返回瓦片，以便在屏幕上呈现
	      return tile;
	    }
	  }]);
	  return DesignTileLayer;
	}(L.TileLayer);

	/**
	 * constructor for Leaflet CoordSys
	 * @param {L.map} map
	 * @param {Object} api
	 */
	function LeafletCoordSys(map, api) {
	  this._map = map;
	  this.dimensions = ['lng', 'lat'];
	  this._mapOffset = [0, 0];
	  this._api = api;
	  this._projection = L.Projection.Mercator;
	}

	LeafletCoordSys.prototype.dimensions = ['lng', 'lat'];

	LeafletCoordSys.prototype.setZoom = function (zoom) {
	  this._zoom = zoom;
	};

	LeafletCoordSys.prototype.setCenter = function (center) {
	  this._center = this._projection.project(new L.LatLng(center[1], center[0]));
	};

	LeafletCoordSys.prototype.setMapOffset = function (mapOffset) {
	  this._mapOffset = mapOffset;
	};

	LeafletCoordSys.prototype.getLeaflet = function () {
	  return this._map;
	};

	LeafletCoordSys.prototype.dataToPoint = function (data) {
	  var point = new L.LatLng(data[1], data[0]);
	  var px = this._map.latLngToLayerPoint(point);
	  var mapOffset = this._mapOffset;
	  return [px.x - mapOffset[0], px.y - mapOffset[1]];
	};

	LeafletCoordSys.prototype.pointToData = function (pt) {
	  var mapOffset = this._mapOffset;
	  var coord = this._map.layerPointToLatLng({
	    x: pt[0] + mapOffset[0],
	    y: pt[1] + mapOffset[1]
	  });
	  return [coord.lng, coord.lat];
	};

	LeafletCoordSys.prototype.convertToPixel = echarts.util.curry(doConvert, 'dataToPoint');

	LeafletCoordSys.prototype.convertFromPixel = echarts.util.curry(doConvert, 'pointToData');

	// ECharts 5 custom series: VR only lists built-in coord systems, so api.coord() would
	// be undefined for 'leaflet'. prepareCustoms() is the extension point ECharts checks
	// first; returning api.coord here wires up coordinate conversion for renderItem.
	LeafletCoordSys.prototype.prepareCustoms = function () {
	  var coordSys = this;
	  var rect = this.getViewRect();
	  return {
	    coordSys: {
	      type: 'leaflet',
	      x: rect.x,
	      y: rect.y,
	      width: rect.width,
	      height: rect.height
	    },
	    api: {
	      coord: function (data) {
	        return coordSys.dataToPoint(data);
	      }
	    }
	  };
	};

	/**
	 * find appropriate coordinate system to convert
	 * @param {*} methodName
	 * @param {*} ecModel
	 * @param {*} finder
	 * @param {*} value
	 * @return {*} converted value
	 */
	function doConvert(methodName, ecModel, finder, value) {
	  var leafletModel = finder.leafletModel;
	  var seriesModel = finder.seriesModel;

	  var coordSys = leafletModel ? leafletModel.coordinateSystem : seriesModel ? seriesModel.coordinateSystem || // For map.
	  (seriesModel.getReferringComponents('leaflet')[0] || {}).coordinateSystem : null;
	  /* eslint-disable no-invalid-this */
	  return coordSys === this ? coordSys[methodName](value) : null;
	}

	LeafletCoordSys.prototype.getViewRect = function () {
	  var api = this._api;
	  return new echarts.graphic.BoundingRect(0, 0, api.getWidth(), api.getHeight());
	};

	LeafletCoordSys.prototype.getRoamTransform = function () {
	  return echarts.matrix.create();
	};

	LeafletCoordSys.dimensions = LeafletCoordSys.prototype.dimensions;

	var CustomOverlay = L.Layer.extend({
	  initialize: function initialize(container) {
	    this._container = container;
	  },

	  onAdd: function onAdd(map) {
	    var pane = map.getPane(this.options.pane);
	    pane.appendChild(this._container);

	    // Calculate initial position of container with
	    // `L.Map.latLngToLayerPoint()`, `getPixelOrigin()
	    // and/or `getPixelBounds()`

	    // L.DomUtil.setPosition(this._container, point);

	    // Add and position children elements if needed

	    // map.on('zoomend viewreset', this._update, this);
	  },

	  onRemove: function onRemove(map) {
	    L.DomUtil.remove(this._container);
	    // map.off('zoomend viewreset', this._update, this);
	  },

	  _update: function _update() {
	    // Recalculate position of container
	    // L.DomUtil.setPosition(this._container, point);
	    // Add/remove/reposition children elements if needed
	  }
	});

	LeafletCoordSys.create = function (ecModel, api) {
	  var leafletCoordSys = void 0;
	  var leafletList = [];
	  var root = api.getDom();

	  // TODO Dispose
	  ecModel.eachComponent('leaflet', function (leafletModel) {
	    var viewportRoot = api.getZr().painter.getViewportRoot();
	    if (typeof L === 'undefined') {
	      throw new Error('Leaflet api is not loaded');
	    }
	    if (leafletCoordSys) {
	      throw new Error('Only one leaflet component can exist');
	    }
	    if (!leafletModel.__map) {
	      // Not support IE8
	      var mapRoot = root.querySelector('.ec-extension-leaflet');
	      if (mapRoot) {
	        // Reset viewport left and top, which will be changed
	        // in moving handler in LeafletView
	        viewportRoot.style.left = '0px';
	        viewportRoot.style.top = '0px';
	        root.removeChild(mapRoot);
	      }
	      mapRoot = document.createElement('div');
	      mapRoot.style.cssText = 'width:100%;height:100%';
	      // Not support IE8
	      mapRoot.classList.add('ec-extension-leaflet');
	      root.appendChild(mapRoot);
	      var _map = leafletModel.__map = L.map(mapRoot, leafletModel.get('leafletOption'));
	      var tiles = leafletModel.get('tiles');
	      var baseLayers = {};
	      var baseLayerAdded = false;
	      var _iteratorNormalCompletion = true;
	      var _didIteratorError = false;
	      var _iteratorError = undefined;

	      try {
	        for (var _iterator = tiles[Symbol.iterator](), _step; !(_iteratorNormalCompletion = (_step = _iterator.next()).done); _iteratorNormalCompletion = true) {
	          var tile = _step.value;

	          // let tileLayer = L.tileLayer(tile.urlTemplate, tile.options);
	          var tileLayer = new DesignTileLayer(tile.urlTemplate, tile.options);
	          if (tile.label) {
	            // only add one baseLayer
	            if (!baseLayerAdded) {
	              tileLayer.addTo(_map);
	              baseLayerAdded = true;
	            }
	            baseLayers[tile.label] = tileLayer;
	          } else {
	            // add all tiles without labels into the map
	            tileLayer.addTo(_map);
	          }
	        }
	        // add layer control when there are more than two layers
	      } catch (err) {
	        _didIteratorError = true;
	        _iteratorError = err;
	      } finally {
	        try {
	          if (!_iteratorNormalCompletion && _iterator.return) {
	            _iterator.return();
	          }
	        } finally {
	          if (_didIteratorError) {
	            throw _iteratorError;
	          }
	        }
	      }

	      if (tiles.length > 1) {
	        var layerControlOpts = leafletModel.get('layerControl');
	        L.control.layers(baseLayers, {}, layerControlOpts).addTo(_map);
	      }

	      /*
	       Encapsulate viewportRoot element into
	       the parent element responsible for moving,
	       avoiding direct manipulation of viewportRoot elements
	       affecting related attributes such as offset.
	      */
	      var moveContainer = document.createElement('div');
	      moveContainer.style = 'position: relative;';
	      moveContainer.appendChild(viewportRoot);

	      new CustomOverlay(moveContainer).addTo(_map);
	    }
	    var map = leafletModel.__map;

	    // Set leaflet options
	    // centerAndZoom before layout and render
	    var center = leafletModel.get('center');
	    var zoom = leafletModel.get('zoom');
	    if (center && zoom) {
	      map.setView([center[1], center[0]], zoom);
	    }

	    leafletCoordSys = new LeafletCoordSys(map, api);
	    leafletList.push(leafletCoordSys);
	    leafletCoordSys.setMapOffset(leafletModel.__mapOffset || [0, 0]);
	    leafletCoordSys.setZoom(zoom);
	    leafletCoordSys.setCenter(center);

	    leafletModel.coordinateSystem = leafletCoordSys;
	  });

	  ecModel.eachSeries(function (seriesModel) {
	    if (seriesModel.get('coordinateSystem') === 'leaflet') {
	      seriesModel.coordinateSystem = leafletCoordSys;
	    }
	  });

	  return leafletList;
	};

	/**
	 * compare if two arrays of length 2 are equal
	 * @param {Array} a array of length 2
	 * @param {Array} b array of length 2
	 * @return {Boolean}
	 */
	function v2Equal(a, b) {
	  return a && b && a[0] === b[0] && a[1] === b[1];
	}

	echarts.extendComponentModel({
	  type: 'leaflet',

	  getLeaflet: function getLeaflet() {
	    // __map is injected when creating LeafletCoordSys
	    return this.__map;
	  },

	  setCenterAndZoom: function setCenterAndZoom(center, zoom) {
	    this.option.center = center;
	    this.option.zoom = zoom;
	  },

	  centerOrZoomChanged: function centerOrZoomChanged(center, zoom) {
	    var option = this.option;
	    return !(v2Equal(center, option.center) && zoom === option.zoom);
	  },

	  defaultOption: {
	    center: [104.114129, 37.550339],
	    zoom: 2,
	    // 除了center和zoom以外的其他leaflet地图选项，都放在这里
	    leafletOption: {
	      zoomControl: false
	    },
	    roam: false,
	    layerControl: {},
	    tiles: [{
	      urlTemplate: 'http://{s}.tile.osm.org/{z}/{x}/{y}.png',
	      options: {
	        attribution: '&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors'
	      }
	    }]
	  }
	});

	echarts.extendComponentView({
	  type: 'leaflet',

	  render: function render(leafletModel, ecModel, api) {
	    var rendering = true;

	    var leaflet = leafletModel.getLeaflet();
	    var moveContainer = api.getZr().painter.getViewportRoot().parentNode;
	    var coordSys = leafletModel.coordinateSystem;

	    var moveHandler = function moveHandler(type, target) {
	      if (rendering) {
	        return;
	      }
	      var offsetEl = leaflet._mapPane;
	      // calculate new mapOffset
	      var transformStyle = offsetEl.style.transform;
	      var dx = 0;
	      var dy = 0;
	      if (transformStyle) {
	        transformStyle = transformStyle.replace('translate3d(', '');
	        var parts = transformStyle.split(',');
	        dx = -parseInt(parts[0], 10);
	        dy = -parseInt(parts[1], 10);
	      } else {
	        // browsers that don't support transform: matrix
	        dx = -parseInt(offsetEl.style.left, 10);
	        dy = -parseInt(offsetEl.style.top, 10);
	      }
	      var mapOffset = [dx, dy];
	      moveContainer.style.left = mapOffset[0] + 'px';
	      moveContainer.style.top = mapOffset[1] + 'px';

	      coordSys.setMapOffset(mapOffset);
	      leafletModel.__mapOffset = mapOffset;

	      api.dispatchAction({
	        type: 'leafletRoam'
	      });
	    };

	    /**
	     * handler for map zoomEnd event
	     */
	    function zoomEndHandler() {
	      if (rendering) return;
	      api.dispatchAction({
	        type: 'leafletRoam'
	      });
	    }

	    /**
	     * handler for map zoom event
	     */
	    function zoomHandler() {
	      moveHandler();
	    }

	    if (this._oldMoveHandler) {
	      leaflet.off('move', this._oldMoveHandler);
	    }
	    if (this._oldZoomHandler) {
	      leaflet.off('zoom', this._oldZoomHandler);
	    }
	    if (this._oldZoomEndHandler) {
	      leaflet.off('zoomend', this._oldZoomEndHandler);
	    }

	    leaflet.on('move', moveHandler);
	    leaflet.on('zoom', zoomHandler);
	    leaflet.on('zoomend', zoomEndHandler);

	    this._oldMoveHandler = moveHandler;
	    this._oldZoomHandler = zoomHandler;
	    this._oldZoomEndHandler = zoomEndHandler;

	    var roam = leafletModel.get('roam');
	    // can move
	    if (roam && roam !== 'scale') {
	      leaflet.dragging.enable();
	    } else {
	      leaflet.dragging.disable();
	    }
	    // can zoom (may need to be more fine-grained)
	    if (roam && roam !== 'move') {
	      leaflet.scrollWheelZoom.enable();
	      leaflet.doubleClickZoom.enable();
	      leaflet.touchZoom.enable();
	    } else {
	      leaflet.scrollWheelZoom.disable();
	      leaflet.doubleClickZoom.disable();
	      leaflet.touchZoom.disable();
	    }

	    rendering = false;
	  }
	});

	/**
	 * Leftlet component extension
	 */


	echarts.registerCoordinateSystem('leaflet', LeafletCoordSys);

	echarts.registerAction({
	  type: 'leafletRoam',
	  event: 'leafletRoam',
	  update: 'updateLayout'
	}, function (payload, ecModel) {
	  ecModel.eachComponent('leaflet', function (leafletModel) {
	    var leaflet = leafletModel.getLeaflet();
	    var center = leaflet.getCenter();
	    leafletModel.setCenterAndZoom([center.lng, center.lat], leaflet.getZoom());
	  });
	});

	var version = '1.0.0';

	exports.version = version;

}));
