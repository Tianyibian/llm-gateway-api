// Deterministic DOM-boundary tests, not a browser or live-model evaluation.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.hidden = false; this.value = ''; this.style = {}; this.classList = { add() {}, remove() {} }; }
  set textContent(value) { this.text = String(value); this.children = []; }
  get textContent() { return (this.text ?? '') + this.children.map(x => x.textContent).join(' '); }
  set innerHTML(value) { throw new Error('Do not render untrusted HTML'); }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.text = ''; this.children = children; }
  addEventListener() {}
  setAttribute() {}
}

function runtime() {
  const nodes = new Map();
  const document = { createElement: tag => new Element(tag), querySelector: key => {
    if (!nodes.has(key)) nodes.set(key, new Element('div'));
    return nodes.get(key);
  } };
  const context = vm.createContext({ document, console, Set });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../../app/static/graph-inspector.js'), 'utf8'), context);
  return { context, inspector: context.GraphInspector, nodes };
}

test('No execution history is invented before receiving events', () => {
  const { inspector } = runtime();
  const view = inspector.create();
  assert.equal(view.panel.hidden, true);
  assert.equal(view.overview.hidden, true);
  assert.equal(view.events.children.length, 0);
});

test('Hierarchical trace distinguishes subagents, tools and maps without rendering HTML', () => {
  const { inspector } = runtime();
  const view = inspector.create();
  inspector.record(view, 'Subagent', {agent_run_id: 'A1.1', agent: 'catalog_agent', stage: 'started', question: '<script>task</script>'});
  inspector.record(view, 'Supervisor', {status: 'complete', agent_runs: [
    {agent_run_id: 'A1.1', agent: 'catalog_agent', status: 'complete', question: 'Supplier?', evidence_ids: ['E1'], tool_calls: 1},
    {agent_run_id: 'A1.2', agent: 'reviews_agent', status: 'complete', question: 'Reviews?', evidence_ids: ['E2'], tool_calls: 1}],
    trace: [{task_id: 'A1.1/R1T1', tool: 'neo4j_relationships', agent: 'catalog_agent', agent_run_id: 'A1.1', question: 'Supplier?'}]});
  assert.match(view.events.textContent, /Business subagents/);
  assert.match(view.events.textContent, /Chosen by: catalog_agent/);
  assert.match(view.overview.textContent, /catalog_agent, reviews_agent/);
  assert.match(view.overview.textContent, /Map tasks are evidence-processing calls/);
  assert.doesNotMatch(view.overview.textContent, /not separate autonomous agents/);
});

test('Answer generation displays real progress and preserves the worker overview', () => {
  const { inspector } = runtime();
  const view = inspector.create();
  inspector.record(view, 'Router', {route: 'graph_rag_search'});
  const overview = view.overview.textContent;
  inspector.record(view, 'Answer generation', {stage: 'map', completed: 2, total: 3});
  inspector.record(view, 'Answer generation', {stage: 'validate_citations'});
  assert.match(view.events.textContent, /Evidence: 2\/3/);
  assert.match(view.events.textContent, /validate_citations/);
  assert.equal(view.overview.textContent, overview);
});

test('Supervisor exposes answer-generation status and passed query checks', () => {
  const { inspector } = runtime();
  const view = inspector.create();
  inspector.record(view, 'Supervisor', {status: 'complete',
    answer_generation: {status: 'complete', mapped_evidence: 2, cited_evidence_ids: ['E1']},
    evidence: [{evidence_id: 'E1', text: 'Evidence', execution: {
      query_mode: 'text_to_cypher', parameters: {}, checks: ['question_alignment', 'neo4j_explain_read_only']}}]});
  assert.match(view.events.textContent, /MapReduce: complete/);
  assert.match(view.events.textContent, /Mapped: 2/);
  assert.match(view.events.textContent, /Checks passed: question_alignment, neo4j_explain_read_only/);
});

test('Neo4j evidence shows the actual query strategy and safe parameter text', () => {
  const { inspector } = runtime();
  const view = inspector.create();
  inspector.record(view, 'Supervisor', {status: 'complete', evidence: [{evidence_id: 'E1', tool: 'neo4j_relationships',
    source_id: 'neo4j:test', text: 'Rows', execution: {query_mode: 'template', template_id: 'product_supplier',
      cypher: 'MATCH (p:Product) RETURN p', parameters: {name: '<script>bad</script>'}, row_count: 1, truncated: false}}]});
  for (const expected of ['template', 'product_supplier', 'MATCH (p:Product)', '<script>bad</script>', 'Rows: 1']) {
    assert.ok(view.events.textContent.includes(expected));
  }
});

test('Router records the actual selected branch, including non-graph routes', () => {
  const { inspector } = runtime();
  const view = inspector.create();
  inspector.record(view, 'Router', { route: 'general_search', reason: 'General question' });
  assert.match(view.events.textContent, /general_search/);
  assert.doesNotMatch(view.events.textContent, /Supervisor|Guardrail/);
  assert.match(view.overview.textContent, /General assistant/);
  assert.match(view.overview.textContent, /does not use the GraphRAG supervisor/);
});

for (const action of ['reject', 'clarify', 'unavailable']) {
  test(`Guardrail ${action} shows a stopped branch without invented tasks`, () => {
    const { inspector } = runtime();
    const view = inspector.create();
    inspector.record(view, 'Guardrail', { action, reason_code: 'test_reason' });
    assert.match(view.events.textContent, /stopped before planning or retrieval/);
    assert.equal(view.events.children.length, 1);
    assert.match(view.overview.textContent, /workers were not started/);
  });
}

test('Supervisor displays actual rounds, dependencies, source IDs and selection', () => {
  const { inspector } = runtime();
  const view = inspector.create();
  inspector.record(view, 'Supervisor', {
    status: 'complete', rounds: 2, tool_calls: 3, reason_code: 'evidence_selected',
    trace: [{ task_id: 'R2T1', tool: 'ms_local_search', question: 'Follow-up question', parent_evidence_ids: ['E1'], evidence_count: 1 }],
    evidence: [{ evidence_id: 'E2', tool: 'ms_local_search', source_id: 'synthetic:test', text: '<img src=x onerror=alert(1)>' }],
    answer_evidence_ids: ['E2'],
  });
  const text = view.events.textContent;
  for (const expected of ['2 retrieval rounds', '3 tool calls', 'R2T1', 'Depends on: E1', 'synthetic:test', 'used in answer', '<img src=x onerror=alert(1)>']) {
    assert.ok(text.includes(expected));
  }
  function tags(node) { return [node.tag, ...node.children.flatMap(tags)]; }
  assert.ok(!tags(view.panel).includes('img'));
  assert.match(view.overview.textContent, /Microsoft Local Search worker/);
  assert.doesNotMatch(view.overview.textContent, /Neo4j relationship worker/);
});

test('Unavailable supervisor shows zero tasks without claiming completion', () => {
  const { inspector } = runtime();
  const view = inspector.create();
  inspector.record(view, 'Supervisor', { status: 'unavailable', reason_code: 'tools_not_connected' });
  assert.match(view.events.textContent, /No retrieval task trace was returned/);
  assert.match(view.events.textContent, /unavailable · 0 retrieval rounds · 0 tool calls/);
  assert.match(view.overview.textContent, /No retrieval workers were reported/);
});

test('Graph routing alone does not claim a worker was called', () => {
  const { inspector } = runtime();
  const view = inspector.create();
  inspector.record(view, 'Router', { route: 'graph_rag_search' });
  assert.match(view.overview.textContent, /No retrieval worker has been reported yet/);
  assert.doesNotMatch(view.overview.textContent, /Microsoft Local Search worker/);
});

test('Worker badges deduplicate actual calls and preserve failure status', () => {
  const { inspector } = runtime();
  const view = inspector.create();
  inspector.record(view, 'Supervisor', {
    status: 'partial', reason_code: 'retrieval_failed', rounds: 2, tool_calls: 3,
    trace: [
      { task_id: 'R1T1', tool: 'ms_local_search', question: 'First', evidence_count: 1 },
      { task_id: 'R2T1', tool: 'ms_local_search', question: 'Next', evidence_count: 1 },
      { task_id: 'R2T2', tool: 'ms_global_search', question: 'Themes', evidence_count: 0, error: 'retrieval_failed' },
    ],
  });
  assert.equal(view.overview.textContent.split('Microsoft Local Search worker').length - 1, 1);
  assert.match(view.overview.textContent, /Microsoft Global Search worker/);
  assert.match(view.overview.textContent, /supervisor: partial/);
  assert.match(view.events.textContent, /Error: retrieval_failed/);
});
