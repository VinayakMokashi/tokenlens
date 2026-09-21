/* Chart helpers for the tokenlens dashboard.
 *
 * Conventions: one axis per chart, thin marks, hairline solid gridlines,
 * a single series colour unless series identity is the point, tooltips on
 * every mark, and colours read from CSS tokens so dark mode is honoured.
 */
(function () {
  "use strict";

  function token(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function fmtTokens(v) {
    if (v >= 1e6) return (v / 1e6).toFixed(2) + "M";
    if (v >= 1e4) return Math.round(v / 1e3) + "K";
    if (v >= 1e3) return (v / 1e3).toFixed(1) + "K";
    return String(v);
  }

  function fmtMoney(v) {
    var a = Math.abs(v);
    if (a >= 1000) return "$" + Math.round(v).toLocaleString();
    if (a >= 1) return "$" + v.toFixed(2);
    if (a >= 0.01) return "$" + v.toFixed(3);
    if (a === 0) return "$0.00";
    return "$" + v.toFixed(4);
  }

  function baseOptions(yFormatter) {
    var ink2 = token("--ink-2"), muted = token("--muted"), grid = token("--grid"), axis = token("--axis");
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: token("--surface"),
          titleColor: token("--ink"),
          bodyColor: ink2,
          borderColor: token("--border"),
          borderWidth: 1,
          padding: 10,
          displayColors: false
        }
      },
      scales: {
        x: {
          grid: { display: false },
          border: { color: axis },
          ticks: { color: muted, maxRotation: 0, autoSkip: true, maxTicksLimit: 12, font: { size: 11 } }
        },
        y: {
          beginAtZero: true,
          grid: { color: grid, lineWidth: 1 },
          border: { display: false },
          ticks: { color: muted, callback: yFormatter, font: { size: 11 }, maxTicksLimit: 6 }
        }
      }
    };
  }

  var registry = {};

  function draw(id, build) {
    var canvas = document.getElementById(id);
    if (!canvas || typeof Chart === "undefined") return;
    if (registry[id]) registry[id].destroy();
    registry[id] = new Chart(canvas.getContext("2d"), build());
  }

  /* Vertical hairlines at compaction turns (context chart only). */
  var compactionPlugin = {
    id: "tlCompactions",
    afterDatasetsDraw: function (chart, args, opts) {
      var idx = opts.indexes || [];
      if (!idx.length) return;
      var ctx = chart.ctx, x = chart.scales.x, area = chart.chartArea;
      ctx.save();
      ctx.strokeStyle = token("--muted");
      ctx.fillStyle = token("--muted");
      ctx.lineWidth = 1;
      ctx.font = "11px " + token("--font");
      idx.forEach(function (i) {
        var px = x.getPixelForValue(i);
        ctx.beginPath(); ctx.moveTo(px, area.top); ctx.lineTo(px, area.bottom); ctx.stroke();
        ctx.fillText("compaction", px + 4, area.top + 12);
      });
      ctx.restore();
    }
  };

  window.tlDailyChart = function (id, daily) {
    draw(id, function () {
      var opts = baseOptions(fmtMoney);
      opts.plugins.tooltip.callbacks = {
        label: function (c) { return fmtMoney(c.parsed.y) + " over " + daily[c.dataIndex].turns + " API calls"; }
      };
      return {
        type: "bar",
        data: {
          labels: daily.map(function (d) { return d.day; }),
          datasets: [{
            data: daily.map(function (d) { return d.cost; }),
            backgroundColor: token("--series-1"),
            maxBarThickness: 24,
            borderRadius: { topLeft: 4, topRight: 4, bottomLeft: 0, bottomRight: 0 },
            borderSkipped: "bottom",
            categoryPercentage: 0.8,
            barPercentage: 0.9
          }]
        },
        options: opts
      };
    });
  };

  window.tlContextChart = function (id, points) {
    draw(id, function () {
      var opts = baseOptions(fmtTokens);
      opts.plugins.tooltip.callbacks = {
        title: function (items) { var p = points[items[0].dataIndex]; return "Turn " + p.turn + (p.model ? " (" + p.model.replace("claude-", "") + ")" : ""); },
        label: function (c) {
          var p = points[c.dataIndex];
          return [fmtTokens(p.context) + " tokens of context", fmtMoney(p.cost) + " this call, " + fmtMoney(p.carry) + " of it carrying context"];
        }
      };
      opts.plugins.tlCompactions = {
        indexes: points.map(function (p, i) { return p.compaction ? i : -1; }).filter(function (i) { return i >= 0; })
      };
      opts.scales.x.title = { display: true, text: "API call", color: token("--muted"), font: { size: 11 } };
      return {
        type: "line",
        data: {
          labels: points.map(function (p) { return p.turn; }),
          datasets: [{
            data: points.map(function (p) { return p.context; }),
            borderColor: token("--series-1"),
            backgroundColor: token("--series-1-soft"),
            borderWidth: 2,
            fill: true,
            tension: 0,
            pointRadius: points.length > 120 ? 0 : 3,
            pointHoverRadius: 5,
            pointBackgroundColor: token("--series-1"),
            pointBorderColor: token("--surface"),
            pointBorderWidth: 2
          }]
        },
        options: opts,
        plugins: [compactionPlugin]
      };
    });
  };

  window.tlTurnCostChart = function (id, points) {
    draw(id, function () {
      var opts = baseOptions(fmtMoney);
      opts.plugins.tooltip.callbacks = {
        title: function (items) { return "Turn " + points[items[0].dataIndex].turn; },
        label: function (c) { var p = points[c.dataIndex]; return fmtMoney(p.cost) + " (" + fmtMoney(p.carry) + " carrying context)"; }
      };
      opts.scales.x.title = { display: true, text: "API call", color: token("--muted"), font: { size: 11 } };
      return {
        type: "bar",
        data: {
          labels: points.map(function (p) { return p.turn; }),
          datasets: [{
            data: points.map(function (p) { return p.cost; }),
            backgroundColor: token("--series-1"),
            maxBarThickness: 24,
            borderRadius: { topLeft: 2, topRight: 2, bottomLeft: 0, bottomRight: 0 },
            borderSkipped: "bottom",
            categoryPercentage: 1.0,
            barPercentage: 0.9
          }]
        },
        options: opts
      };
    });
  };

  /* Re-draw with the new tokens when the colour scheme flips. */
  if (window.matchMedia) {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () {
      var d = window.TL_DATA || {};
      if (d.daily) tlDailyChart("daily-chart", d.daily);
      if (d.context) { tlContextChart("context-chart", d.context); tlTurnCostChart("turn-cost-chart", d.context); }
    });
  }
})();
