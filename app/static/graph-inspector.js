/* Render only server-reported events. Never interpret evidence as HTML. */
(function (root) {
  const workers = {
    ms_local_search: "Microsoft Local Search worker",
    ms_global_search: "Microsoft Global Search worker",
    ms_drift_search: "Microsoft DRIFT Search worker",
    neo4j_relationships: "Neo4j relationship worker",
  };
  const branches = {
    general_search: "General assistant",
    product_search: "Product catalog branch",
    additional_search: "Clarification branch",
    policy_search: "Policy and support RAG branch",
    analytics_search: "Snowflake analytics branch",
    graph_rag_search: "GraphRAG branch",
    vision_analysis: "Image analysis branch",
  };

  function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = String(text);
    if (className) node.className = className;
    return node;
  }

  function create() {
    const overview = element("div", undefined, "execution-overview");
    overview.hidden = true;
    overview.setAttribute("aria-label", "Selected branch and retrieval workers");
    const panel = element("details", undefined, "execution-panel");
    panel.hidden = true;
    panel.append(element("summary", "Execution details"));
    const events = element("div", undefined, "execution-events");
    panel.append(events);
    return { panel, events, overview };
  }

  function record(inspector, name, payload) {
    inspector.panel.hidden = false;
    const section = element("section", undefined, "execution-step");
    section.append(element("strong", name));
    if (name === "Router") {
      inspector.overview.hidden = false;
      inspector.overview.replaceChildren(element("strong", `Handling this request: ${branches[payload.route] ?? payload.route}`));
      inspector.overview.append(element("p", payload.route === "graph_rag_search"
        ? "Waiting for the graph branch result. No retrieval worker has been reported yet."
        : "This route does not use the GraphRAG supervisor.", "execution-note"));
      section.append(element("p", `Selected: ${payload.route}`));
      if (payload.reason) section.append(element("p", payload.reason, "execution-note"));
    } else if (name === "Guardrail") {
      inspector.overview.hidden = false;
      inspector.overview.replaceChildren(element("strong", `GraphRAG scope: ${payload.action}`));
      inspector.overview.append(element("p", payload.action === "allow"
        ? "Scope approved. Waiting for the supervisor result."
        : "Stopped at the scope gate. Supervisor and retrieval workers were not started.", "execution-note"));
      section.append(element("p", `${payload.action} · ${payload.reason_code}`));
      section.append(element("p", payload.action === "allow"
        ? "Scope passed. The supervisor receives the original question."
        : "This branch stopped before planning or retrieval."));
    } else if (name === "Clarification guardrail") {
      section.append(element("p", `Decision: ${payload.action}`));
      if ((payload.missing ?? []).length) section.append(element("p", `Missing: ${payload.missing.join(", ")}`));
      if (payload.action !== "ready") section.append(element("p", "No retrieval tools were started."));
    } else if (name === "Subagent") {
      section.append(element("p", `${payload.agent_run_id} · ${payload.agent} · ${payload.stage}`));
      if (payload.question) section.append(element("p", payload.question));
      if (payload.status) section.append(element("p", `Status: ${payload.status} · Tool calls: ${payload.tool_calls}`));
    } else if (name === "Answer generation") {
      section.append(element("p", `MapReduce stage: ${payload.stage}`));
      if (payload.total !== undefined) section.append(element("p", payload.completed !== undefined
        ? `Evidence: ${payload.completed}/${payload.total}` : `Sources to combine: ${payload.total}`));
      if (payload.reason_code) section.append(element("p", payload.reason_code));
    } else if (name === "Supervisor") {
      inspector.overview.hidden = false;
      inspector.overview.replaceChildren(element("strong", `GraphRAG supervisor: ${payload.status}`));
      const attempted = [...new Set((payload.trace ?? []).map(task => task.tool))];
      const badges = element("div", undefined, "worker-badges");
      for (const tool of attempted) {
        badges.append(element("span", workers[tool] ?? tool, "worker-badge"));
      }
      inspector.overview.append(badges);
      inspector.overview.append(element("p", attempted.length
        ? `${payload.tool_calls ?? 0} tool calls · ${payload.rounds ?? 0} retrieval rounds. Expand Execution details for task results and failures.`
        : "No retrieval workers were reported.", "execution-note"));
      inspector.overview.append(element("p", (payload.agent_runs ?? []).length
        ? "Supervisor delegates business goals; specialist agents independently choose tools. Map tasks are evidence-processing calls, not subagents."
        : "Workers are registered retrieval tools, not separate autonomous agents.", "execution-note"));
      if ((payload.agent_runs ?? []).length) {
        section.append(element("strong", "Business subagents"));
        inspector.overview.append(element("p", `Subagents: ${[...new Set(payload.agent_runs.map(run => run.agent))].join(", ")}`, "execution-note"));
        for (const run of payload.agent_runs) {
          const item = element("div");
          item.append(element("strong", `${run.agent_run_id} · ${run.agent} · ${run.status}`));
          item.append(element("p", run.question));
          item.append(element("p", `Depends on: ${(run.parent_evidence_ids ?? []).join(", ") || "original question"} · Evidence: ${(run.evidence_ids ?? []).join(", ") || "none"} · Tool calls: ${run.tool_calls}`));
          section.append(item);
        }
      }
      section.append(element("p", `${payload.status} · ${payload.rounds ?? 0} retrieval rounds · ${payload.tool_calls ?? 0} tool calls`));
      section.append(element("p", `Outcome: ${payload.reason_code}`, "execution-note"));
      if (payload.answer_generation) {
        const generation = payload.answer_generation;
        section.append(element("p", `MapReduce: ${generation.status} · Mapped: ${generation.mapped_evidence} · Cited: ${(generation.cited_evidence_ids ?? []).join(", ") || "none"}`, "execution-note"));
      }
      const tasks = element("ol", undefined, "execution-tasks");
      for (const task of payload.trace ?? []) {
        const row = element("li");
        row.append(element("strong", `${task.task_id} · ${task.tool}`));
        if (task.agent) row.append(element("p", `Chosen by: ${task.agent} (${task.agent_run_id})`, "execution-note"));
        row.append(element("p", workers[task.tool] ?? task.tool, "execution-note"));
        row.append(element("p", task.question));
        row.append(element("p", `Depends on: ${task.parent_evidence_ids?.join(", ") || "original question"} · Returned: ${task.evidence_count ?? 0} excerpts${task.error ? ` · Error: ${task.error}` : ""}`, "execution-note"));
        tasks.append(row);
      }
      section.append(tasks);
      if (!(payload.trace ?? []).length) section.append(element("p", "No retrieval task trace was returned."));
      if ((payload.evidence ?? []).length) {
        section.append(element("strong", "Retrieved evidence"));
        const selected = new Set(payload.answer_evidence_ids ?? []);
        for (const evidence of payload.evidence) {
          const card = element("details", undefined, "graph-evidence");
          card.append(element("summary", `[${evidence.evidence_id}] ${evidence.tool}${selected.has(evidence.evidence_id) ? " · used in answer" : ""}`));
          card.append(element("p", evidence.source_id, "execution-source-id"));
          card.append(element("p", evidence.text));
          if (evidence.execution) {
            const run = evidence.execution;
            card.append(element("strong", `Neo4j strategy: ${run.query_mode}${run.template_id ? ` · ${run.template_id}` : ""}`));
            card.append(element("pre", run.cypher));
            card.append(element("pre", JSON.stringify(run.parameters, null, 2)));
            card.append(element("p", `Rows: ${run.row_count} · Truncated: ${run.truncated}`, "execution-note"));
            card.append(element("p", `Checks passed: ${(run.checks ?? []).join(", ") || "not reported"}`, "execution-note"));
          }
          section.append(card);
        }
      }
      section.append(element("p", "Completed execution trace, not live internal reasoning. Source IDs do not guarantee answer accuracy.", "execution-note"));
    }
    inspector.events.append(section);
  }

  root.GraphInspector = { create, record };
})(globalThis);
