const fs = require('fs');
const os = require('os');
const path = require('path');

const ROMPTH = /^OMP_NUM_THREADS=(\d+)/m;
const RGRAPH = /^Loading graph .*\/(.+?)\.txt \.\.\./m;
const RORDER = /^order: (\d+) size: (\d+) \[directed\] \{\}/m;
const RRESLT = /^\{\-(.+?)\/\+(.+?) batchf, (.+?) batchi, (.+?) threads, (.+?) frontier, (.+?) prune\} -> \{(.+?)ms, (.+?)ms init, (.+?)ms mark, (.+?)ms comp, (.+?) iter, (.+?) err, (.+?) aff\} (\w+)/m;



// *-FILE
// ------

function readFile(pth) {
  var d = fs.readFileSync(pth, 'utf8');
  return d.replace(/\r?\n/g, '\n');
}

function writeFile(pth, d) {
  d = d.replace(/\r?\n/g, os.EOL);
  fs.writeFileSync(pth, d);
}




// *-CSV
// -----

function writeCsv(pth, rows) {
  var cols = Object.keys(rows[0]);
  var a = cols.join()+'\n';
  for (var r of rows)
    a += [...Object.values(r)].map(v => `"${v}"`).join()+'\n';
  writeFile(pth, a);
}




// *-LOG
// -----

function readLogLine(ln, data, state) {
  ln = ln.replace(/^\d+-\d+-\d+ \d+:\d+:\d+\s+/, '');
  if (ROMPTH.test(ln)) {
    var [, omp_num_threads] = ROMPTH.exec(ln);
    state.omp_num_threads   = parseFloat(omp_num_threads);
  }
  else if (RGRAPH.test(ln)) {
    var [, graph] = RGRAPH.exec(ln);
    if (!data.has(graph)) data.set(graph, []);
    state.graph   = graph;
  }
  else if (RORDER.test(ln)) {
    var [, order, size] = RORDER.exec(ln);
    state.order = parseFloat(order);
    state.size  = parseFloat(size);
  }
  else if (RRESLT.test(ln)) {
    var [,
      batch_deletions_fraction, batch_insertions_fraction, batch_index,
      num_threads, frontier_tolerance, prune_tolerance,
      time, initialization_time, marking_time, computation_time,
      iterations, error, affected_vertices, technique,
    ] = RRESLT.exec(ln);
    data.get(state.graph).push(Object.assign({}, state, {
      batch_deletions_fraction:  parseFloat(batch_deletions_fraction),
      batch_insertions_fraction: parseFloat(batch_insertions_fraction),
      batch_index:               parseFloat(batch_index),
      num_threads:               parseFloat(num_threads),
      frontier_tolerance:        parseFloat(frontier_tolerance),
      prune_tolerance:           parseFloat(prune_tolerance),
      time:                parseFloat(time),
      initialization_time: parseFloat(initialization_time),
      marking_time:        parseFloat(marking_time),
      computation_time:    parseFloat(computation_time),
      iterations:  parseFloat(iterations),
      error:       parseFloat(error),
      affected_vertices: parseFloat(affected_vertices),
      technique,
    }));
  }
  return state;
}

function readLog(pth) {
  var text  = readFile(pth);
  var lines = text.split('\n');
  var data  = new Map();
  var state = {};
  for (var ln of lines)
    state = readLogLine(ln, data, state);
  return data;
}




// PROCESS-*
// ---------

function processCsv(data) {
  var a = [];
  for (var rows of data.values())
    a.push(...rows);
  return a;
}




// MAIN
// ----

function main(cmd, log, out) {
  var data = readLog(log);
  if (path.extname(out)==='') cmd += '-dir';
  switch (cmd) {
    case 'csv':
      var rows = processCsv(data);
      writeCsv(out, rows);
      break;
    case 'csv-dir':
      for (var [graph, rows] of data)
        writeCsv(path.join(out, graph+'.csv'), rows);
      break;
    default:
      console.error(`error: "${cmd}"?`);
      break;
  }
}
main(...process.argv.slice(2));
