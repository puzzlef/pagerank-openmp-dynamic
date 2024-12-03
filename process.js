const fs = require('fs');
const os = require('os');
const path = require('path');

const ROMPTH = /^OMP_NUM_THREADS=(\d+)/m;
const RGRAPH = /^Loading graph .*\/(.+?)\.txt \.\.\./m;
const RORDER = /^order: (\d+) size: (\d+) \[directed\] \{\}/m;
const RITERS = /^Iteration (\d+): (\d+)\/(\d+) affected vertices/m;
const RRESLT = /^\{\-(.+?)\/\+(.+?) batchf, (.+?) batchi, (.+?) threads, (.+?) frontier, (.+?) prune\} -> \{(.+?)ms, (.+?)ms init, (.+?)ms mark, (.+?)ms comp, (.+?) iter, (.+?) err\} (\w+)/m;




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
  var cols = Object.keys(rows[0]).filter(k => k!=='iteration_data');
  var a = cols.join()+'\n';
  for (var r of rows) {
    var line = '';
    for (var c of cols)
      if (c!=='iteration_data') line += `"${r[c]}",`;
    line = line.slice(0, -1);
    a += line+'\n';
  }
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
    state.batch_deletions_fraction  = 0;
    state.batch_insertions_fraction = 0;
    state.batch_index               = 0;
    state.num_threads               = 0;
    state.iteration_data    = [];
    state.iteration_id      = -1;
    state.affected_vertices = 0;
    state.total_vertices    = 0;
  }
  else if (RITERS.test(ln)) {
    var [, iteration_id, affected_vertices, total_vertices] = RITERS.exec(ln);
    if (iteration_id==='0') state.iteration_data = [];
    state.iteration_data.push({
      iteration_id:      parseFloat(iteration_id),
      affected_vertices: parseFloat(affected_vertices),
      total_vertices:    parseFloat(total_vertices),
    });
  }
  else if (RRESLT.test(ln)) {
    var [,
      batch_deletions_fraction, batch_insertions_fraction, batch_index,
      num_threads, frontier_tolerance, prune_tolerance,
      time, initialization_time, marking_time, computation_time,
      iterations, error, technique,
    ] = RRESLT.exec(ln);
    state.batch_deletions_fraction  = parseFloat(batch_deletions_fraction);
    state.batch_insertions_fraction = parseFloat(batch_insertions_fraction);
    state.batch_index               = parseFloat(batch_index);
    state.num_threads               = parseFloat(num_threads);
    state.frontier_tolerance        = parseFloat(frontier_tolerance);
    state.prune_tolerance           = parseFloat(prune_tolerance);
    state.time                = parseFloat(time);
    state.initialization_time = parseFloat(initialization_time);
    state.marking_time        = parseFloat(marking_time);
    state.computation_time    = parseFloat(computation_time);
    state.iterations          = parseFloat(iterations);
    state.error               = parseFloat(error);
    state.technique           = technique;
    var I = state.iteration_data.length;
    data.get(state.graph).push(Object.assign({}, state, {
      iteration_data: [],
      iteration_id:   -1,
      affected_vertices: state.iteration_data.length > 0? state.iteration_data[I-1].affected_vertices : 0,
      total_vertices:    state.iteration_data.length > 0? state.iteration_data[I-1].total_vertices    : state.order,
    }));
    for (var iter of state.iteration_data)
      data.get(state.graph).push(Object.assign({}, state, iter));
    state.iteration_data = [];
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
