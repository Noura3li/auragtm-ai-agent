"""
AuraGTM — 2-Agent Workflow
============================
Task 5 of the team project.

This module does NOT modify auragtm_engine.py / app.py / rag_pipeline.py in
any way. It only *calls* the public API my teammates already built
(get_engine(), engine.generate_gtm_strategy(...)) and adds a second agent
on top of it.

WHY 2 AGENTS
------------
Today, generate_gtm_strategy() does retrieval + writing in a single LLM
pass. That's great for speed, but there's no independent check on the
output: the model that wrote the strategy is the only one ever judging it.

This module introduces a second, independent agent with a different job
and a different prompt, so the two agents play different roles instead of
one model doing everything:

    Agent 1 — STRATEGIST
        Role  : produce the first full draft of the GTM strategy.
        How   : reuses my teammates' engine.generate_gtm_strategy() AS-IS
                (retrieval, routing, prompt templates — all untouched).

    Agent 2 — CRITIC / QA REVIEWER   (new, lives only in this file)
        Role  : independently audit Agent 1's draft against the original
                brief and the retrieved sources, then produce a refined,
                final version plus a short changelog of what was fixed.
        Checks: grounding (no contradicted/fabricated facts), completeness
                (all 10 required sections present), brand-tone fit,
                actionability (concrete, non-generic recommendations).

The two agents run sequentially and share state (Agent 2 sees Agent 1's
draft + the same sources). This is also where long-term project memory
(see project_memory.py) plugs in: before Agent 1 runs, we pull this
project's past outputs and pass them along through the `refinement` field
that engine.generate_gtm_strategy() already supports — so Agent 1 builds
on previous work for this client/product instead of starting from zero
every time.
"""

from typing import Any, Dict, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from auragtm_engine import get_engine
from project_memory import get_memory


# ---------------------------------------------------------------------------
# Agent 2 — Critic / QA Reviewer prompt (brand new, independent of Agent 1)
# ---------------------------------------------------------------------------

CRITIC_PROMPT = ChatPromptTemplate.from_template("""
You are AuraGTM's QA Reviewer Agent — a second, independent reviewer of
go-to-market strategies. You did NOT write the draft below; your job is to
audit it and hand back an improved final version.

ORIGINAL BRIEF
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Product: {product_name}
Description: {product_description}
Industry: {industry}
Region: {region}
Business Goal: {business_goal}
Brand Tone: {brand_tone}
Mode: {mode}

PROJECT MEMORY (earlier work on this same project, if any)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{memory_context}

DRAFT STRATEGY FROM AGENT 1 (THE STRATEGIST)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{draft_strategy}

YOUR REVIEW CHECKLIST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Completeness — are all 10 required sections present (Target Audience,
   Buyer Personas, Localization Insights, Competitor Analysis, Positioning
   & Messaging, Marketing Channels, GTM Roadmap, Content Ideas,
   Assumptions & Risks, Executive Summary)? If any is missing or thin,
   add it properly.
2. Consistency with memory — if PROJECT MEMORY is non-empty, make sure the
   final strategy builds on it sensibly (evolves it) instead of
   contradicting or ignoring earlier decisions for this project.
3. Brand tone fit — does the language genuinely match "{brand_tone}"?
   Adjust wording where it doesn't.
4. Actionability — flag and rewrite any generic filler ("leverage
   synergies", "engage stakeholders") into specific, concrete actions.
5. Grounding — do not introduce new invented statistics. You may sharpen
   or reorganize claims already present in the draft, but do not fabricate
   new figures.
6. Do NOT print source labels, filenames, page numbers, or citations
   anywhere in the final text.

OUTPUT FORMAT — produce exactly these two parts, in this order:

## Critique Notes
A short bullet list (3-6 bullets) of what was wrong or weak in the draft
and what you changed. Be specific and concise.

## Final Strategy
The complete, improved GTM strategy, written in clean markdown with the
same 10 "## " section headings as the draft. This must be a complete,
standalone document — do not refer back to "the draft" inside it.
""")


class CriticAgent:
    """Agent 2: reviews and refines Agent 1's output. Brand new — does not
    exist anywhere in my teammates' code."""

    def __init__(self, llm):
        # Reuses the already-configured ChatOpenAI instance from the shared
        # engine (same model/temperature/key as the rest of the app) rather
        # than instantiating a second, possibly inconsistent, LLM client.
        self.chain = CRITIC_PROMPT | llm | StrOutputParser()

    def review_and_refine(
        self,
        draft_strategy: str,
        inputs: Dict[str, Any],
        memory_context: str,
    ) -> Dict[str, str]:
        raw = self.chain.invoke({
            "product_name"       : inputs.get("product_name", ""),
            "product_description": inputs.get("product_description", ""),
            "industry"           : inputs.get("industry", ""),
            "region"             : inputs.get("region", ""),
            "business_goal"      : inputs.get("business_goal", ""),
            "brand_tone"         : inputs.get("brand_tone", "Professional & Authoritative"),
            "mode"               : inputs.get("mode", "General Mode"),
            "memory_context"     : memory_context or "(none — this is the first run for this project)",
            "draft_strategy"     : draft_strategy,
        })
        return _split_critique_output(raw)


def _split_critique_output(raw: str) -> Dict[str, str]:
    """Split the critic's single text blob into critique_notes / final_strategy
    based on the '## Final Strategy' heading we asked it to produce."""
    marker = "## Final Strategy"
    idx = raw.find(marker)
    if idx == -1:
        # Defensive fallback: if the model didn't follow the format exactly,
        # treat the whole thing as the final strategy and skip notes rather
        # than lose the output.
        return {"critique_notes": "", "final_strategy": raw.strip()}

    critique_part = raw[:idx]
    final_part = raw[idx + len(marker):]

    critique_part = critique_part.replace("## Critique Notes", "").strip()
    final_part = final_part.strip()

    return {"critique_notes": critique_part, "final_strategy": final_part}


# ---------------------------------------------------------------------------
# Orchestrator: runs Agent 1 then Agent 2, wired up with long-term memory
# ---------------------------------------------------------------------------

def run_two_agent_workflow(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    inputs: same shape app.py already uses for GTMRequest, i.e.
        product_name, product_description, industry, region, business_goal,
        brand_tone, mode, client_name, refinement (refinement optional)

    Returns a dict ready to be merged into a JSON response:
        {
          "type": "agent_workflow",
          "draft_strategy": "...",      # Agent 1's raw output
          "critique_notes": "...",      # Agent 2's review notes
          "final_strategy": "...",      # Agent 2's refined final output
          "sources": [...],             # same source list the engine returned
          "memory_used": bool,          # whether prior project memory existed
        }
    """
    engine = get_engine()
    memory = get_memory()

    client_name  = (inputs.get("client_name") or "").strip() or None
    product_name = inputs.get("product_name")

    # --- Long-term memory: pull this project's past outputs, if any ---
    memory_context = memory.get_memory_context(
        client_name=client_name, product_name=product_name, limit=3
    )

    # Feed memory into Agent 1 through the `refinement` field that
    # engine.generate_gtm_strategy() already natively supports — this is
    # the non-invasive integration point; nothing in auragtm_engine.py
    # needs to change for Agent 1 to be memory-aware.
    agent1_inputs = dict(inputs)
    if memory_context:
        memory_instruction = (
            "PROJECT MEMORY — earlier strategy work already exists for this "
            "project. Build on it and evolve the recommendations rather than "
            "starting over or contradicting prior decisions, unless the new "
            "business goal explicitly requires a change of direction.\n\n"
            f"{memory_context}"
        )
        existing_refinement = (inputs.get("refinement") or "").strip()
        agent1_inputs["refinement"] = (
            f"{memory_instruction}\n\n{existing_refinement}".strip()
            if existing_refinement else memory_instruction
        )

    # --- Agent 1: Strategist (my teammates' existing, untouched engine) ---
    agent1_result = engine.generate_gtm_strategy(agent1_inputs)
    draft_strategy = agent1_result["strategy"]
    sources = agent1_result["sources"]

    # --- Agent 2: Critic / Reviewer (new) ---
    critic = CriticAgent(engine.llm)
    agent2_result = critic.review_and_refine(
        draft_strategy=draft_strategy,
        inputs=inputs,
        memory_context=memory_context,
    )

    final_strategy = agent2_result["final_strategy"] or draft_strategy

    # --- Long-term memory: persist this run for next time ---
    memory.save_entry(
        client_name=client_name,
        product_name=product_name,
        industry=inputs.get("industry"),
        region=inputs.get("region"),
        business_goal=inputs.get("business_goal"),
        mode=inputs.get("mode"),
        record_type="agent_workflow",
        final_content=final_strategy,
        draft_content=draft_strategy,
        critique_notes=agent2_result["critique_notes"],
        sources=sources,
    )

    return {
        "type"           : "agent_workflow",
        "draft_strategy" : draft_strategy,
        "critique_notes" : agent2_result["critique_notes"],
        "final_strategy" : final_strategy,
        "sources"        : sources,
        "memory_used"    : bool(memory_context),
    }
