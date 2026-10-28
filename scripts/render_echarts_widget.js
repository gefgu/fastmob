// Draw the chart in a fastmob-vis notebook output as static SVG, using ECharts' server-side renderer.
//
// Usage: node render_echarts_widget.js ECHARTS_JS WIDTH HEIGHT < widget_script.js > chart.svg
// The widget script is the <script> of a `bundle_libs=False` chart (see fastmob_vis.figure): it builds
// the option, applies its formatters and calls `echarts.init(element)`. Here `echarts.init` returns a
// server-side SVG instance, so the chart is drawn exactly as in a browser, without a browser.
const fs = require("fs");

const [echartsPath, width, height] = process.argv.slice(2);
const echarts = require(echartsPath);
const script = fs.readFileSync(0, "utf8");

let instance = null;
const element = {};
// Passed to the widget script only: globals would make ECharts believe it runs in a browser.
const fakeWindow = { addEventListener() {} };
const fakeDocument = { getElementById: () => element };
// ECharts writes `fontFamily` unescaped into an SVG attribute, so a family quoted with " breaks the XML.
const json = {
  parse: (text) =>
    JSON.parse(text, (key, value) =>
      key === "fontFamily" && typeof value === "string" ? value.replace(/"/g, "'") : value,
    ),
  stringify: JSON.stringify,
};
const shim = Object.assign({}, echarts, {
  init: () => {
    instance = echarts.init(null, null, { renderer: "svg", ssr: true, width: Number(width), height: Number(height) });
    return instance;
  },
});
try {
  new Function("echarts", "document", "window", "JSON", script)(shim, fakeDocument, fakeWindow, json);
} catch (error) {
  // ECharts' own frames are minified onto one huge line: keep the message readable.
  console.error(`widget script failed: ${String(error && error.message).slice(0, 300)}`);
  process.exit(1);
}
if (!instance) throw new Error("the widget script did not create a chart");
// Some chart types leave a timer running; exit explicitly once the SVG has been written.
process.stdout.write(instance.renderToSVGString(), () => process.exit(0));
