"""
AuraGTM Ultimate Engine
========================

Features
---------
- Smart Intent Detection
- Strategy Mode
- Ask Aura Mode
- Router-based Retrieval
- Hybrid Retrieval (semantic MMR + BM25 keyword)   <-- merged from Heba's update
- Hybrid Global + Client Knowledge
- Structured GTM Generation (10 sections)
- Strategic Q&A
- Source Tracking
- Inline Source Citations + Justification           <-- Strategy Justifier
- Singleton Pattern
- Per-Client Vector DB with hot-reload support
"""

import os
import re
from itertools import zip_longest
from typing import List, Dict, Any, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from rank_bm25 import BM25Okapi

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

DB_DIRECTORY    = os.path.join(BASE_DIR, "vector_db", "global_db")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL       = "gpt-4o"
TEMPERATURE     = 0
MAX_TOKENS      = 4000   # bumped from 3500: citations + justifications add length

TOPICS = [
    "01_GTM_Frameworks",
    "02_AI_Business_Solutions",
    "03_Market_Knowledge",
    "04_Localization",
    "05_Personas",
    "06_Technology_and_Digital_Capabilities",
    "07_Competitor_Knowledge",
]

CLIENT_TOPIC = "08_Clients_Data"
K_PER_TOPIC  = 3
CLIENT_K     = 5


def bm25_tokenize(text: str) -> list:
    """Tokenizer for BM25 keyword search. Handles English + Arabic."""
    text = (text or "").lower()
    return re.findall(r"[a-zA-Z0-9\u0600-\u06FF]+", text)


class IntentDecision(BaseModel):
    """Detect the type of request."""
    intent: Literal[
        "strategy_generation",
        "question_answering",
    ] = Field(
        description=(
            "strategy_generation = build a complete GTM strategy. "
            "question_answering = answer a strategic question."
        )
    )
    reasoning: str


class RouteDecision(BaseModel):
    """Select relevant knowledge domains."""
    domains: List[
        Literal[
            "01_GTM_Frameworks",
            "02_AI_Business_Solutions",
            "03_Market_Knowledge",
            "04_Localization",
            "05_Personas",
            "06_Technology_and_Digital_Capabilities",
            "07_Competitor_Knowledge",
        ]
    ] = Field(description="Knowledge domains required.")
    reasoning: str


class AuraGTMEngine:

    def __init__(self):
        print("Initializing AuraGTM Ultimate Engine...")

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is missing.")

        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

        self.vector_db = Chroma(
            persist_directory=DB_DIRECTORY,
            embedding_function=self.embeddings,
        )

        # Build the BM25 keyword index from the global DB (for hybrid retrieval).
        self._build_bm25_index()

        # Cache of client_name -> Chroma instance
        self.client_dbs: Dict[str, Chroma] = {}

        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

        self.available_clients = self._get_available_clients()
        self._setup_intent_detector()
        self._setup_router()

        print(f"Engine ready. Clients found: {self.available_clients}")

    # ------------------------------------------------------------------
    # Client management
    # ------------------------------------------------------------------

    def _get_available_clients(self) -> List[str]:
        clients_folder = os.path.join(BASE_DIR, "vector_db", "clients")
        if not os.path.exists(clients_folder):
            return []
        clients = []
        for entry in os.listdir(clients_folder):
            if entry.endswith("_db") and os.path.isdir(
                os.path.join(clients_folder, entry)
            ):
                clients.append(entry.replace("_db", ""))
        return sorted(clients)

    def reload_client(self, client_name: str):
        """
        Call this after a new upload to:
        1. Evict the stale cached Chroma handle for this client (if any).
        2. Refresh the available_clients list so the new client appears in the UI.
        """
        if client_name in self.client_dbs:
            del self.client_dbs[client_name]
        self.available_clients = self._get_available_clients()
        print(f"Engine reloaded for client: {client_name}")

    # ------------------------------------------------------------------
    # LLM chains
    # ------------------------------------------------------------------

    def _setup_intent_detector(self):
        system_prompt = """
You classify user requests.

Return:

strategy_generation:
- build GTM plans
- launch strategies
- market entry strategies
- positioning exercises
- complete strategic outputs

question_answering:
- answer questions
- explain concepts
- compare competitors
- provide recommendations
- quick strategic advice
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{query}")
        ])
        self.intent_chain = prompt | self.llm.with_structured_output(IntentDecision)

    def _setup_router(self):
        router_system = """
You are AuraGTM's retrieval router.

Choose ONLY the knowledge domains required to answer the request.
Avoid selecting unnecessary domains.

Knowledge domains:

01_GTM_Frameworks:
GTM frameworks and launch structures.

02_AI_Business_Solutions:
AI business use cases and automation.

03_Market_Knowledge:
Markets, trends, opportunities.

04_Localization:
Regional adaptation and regulations.

05_Personas:
Customer segments and pain points.

06_Technology_and_Digital_Capabilities:
Technology capabilities.

07_Competitor_Knowledge:
Competitor analysis.
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", router_system),
            ("human", "{query}")
        ])
        self.router_chain = prompt | self.llm.with_structured_output(RouteDecision)

    # ------------------------------------------------------------------
    # Query building
    # ------------------------------------------------------------------

    def _build_strategy_query(self, inputs: Dict[str, Any]) -> str:
        return (
            f"{inputs.get('product_name', '')} "
            f"{inputs.get('product_description', '')} "
            f"{inputs.get('industry', '')} "
            f"{inputs.get('region', '')} "
            f"{inputs.get('business_goal', '')}"
        ).strip()

    def _detect_intent(self, query: str) -> IntentDecision:
        try:
            return self.intent_chain.invoke({"query": query})
        except Exception:
            return IntentDecision(
                intent="question_answering",
                reasoning="Fallback intent."
            )

    def _route_query(self, query: str) -> RouteDecision:
        try:
            return self.router_chain.invoke({"query": query})
        except Exception:
            return RouteDecision(domains=TOPICS, reasoning="Fallback routing.")

    # ------------------------------------------------------------------
    # BM25 keyword index (hybrid retrieval)
    # ------------------------------------------------------------------

    def _build_bm25_index(self):
        """
        Build a BM25 keyword index from the existing global ChromaDB documents.
        This does NOT rebuild or change ChromaDB; it only reads it.
        """
        try:
            db_data = self.vector_db.get(include=["documents", "metadatas"])
            self.bm25_documents = db_data.get("documents", []) or []
            self.bm25_metadatas = db_data.get("metadatas", []) or []

            tokenized_docs = [bm25_tokenize(doc) for doc in self.bm25_documents]

            if tokenized_docs:
                self.bm25_index = BM25Okapi(tokenized_docs)
                print(f"BM25 index ready. Documents indexed: {len(self.bm25_documents)}")
            else:
                self.bm25_index = None
                print("BM25 index skipped — no documents found in global DB.")

        except Exception as e:
            # On failure, disable BM25 gracefully (semantic search still works).
            print(f"BM25 index failed: {e}")
            self.bm25_documents = []
            self.bm25_metadatas = []
            self.bm25_index = None

    def _bm25_search_from_topic(
        self, query: str, topic: str, k: int = K_PER_TOPIC
    ) -> List[Document]:
        """Keyword search using BM25, filtered by topic."""
        if self.bm25_index is None:
            return []

        query_tokens = bm25_tokenize(query)
        scores = self.bm25_index.get_scores(query_tokens)

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results: List[Document] = []
        for idx in ranked_indexes:
            metadata = self.bm25_metadatas[idx] or {}

            if metadata.get("topic") != topic:
                continue
            if scores[idx] <= 0:
                continue

            results.append(Document(
                page_content=self.bm25_documents[idx],
                metadata={
                    **metadata,
                    "bm25_score": float(scores[idx]),
                    "retrieval_type": "bm25",
                },
            ))

            if len(results) >= k:
                break

        return results

    # ------------------------------------------------------------------
    # Retrieval (hybrid: semantic MMR + BM25 keyword)
    # ------------------------------------------------------------------

    def _retrieve_from_topic(
        self, query: str, topic: str, k: int = K_PER_TOPIC
    ) -> List[Document]:
        """
        Hybrid retrieval for a single topic:
        semantic (MMR) results + BM25 keyword results, interleaved & deduplicated.
        """
        # 1) Semantic search (MMR) with topic filter
        semantic_results: List[Document] = []
        try:
            semantic_results = self.vector_db.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=max(k * 3, 10),
                filter={"topic": {"$eq": topic}},
            )
            for doc in semantic_results:
                doc.metadata["retrieval_type"] = "semantic"
        except Exception:
            semantic_results = []

        # 2) Keyword search (BM25) with topic filter
        keyword_results = self._bm25_search_from_topic(query, topic, k=k)

        # 3) Interleave + dedup, capped at k (a real mix of both signals)
        combined: List[Document] = []
        seen = set()
        for pair in zip_longest(semantic_results, keyword_results):
            for doc in pair:
                if doc is None:
                    continue
                key = doc.page_content[:300]
                if key in seen:
                    continue
                seen.add(key)
                combined.append(doc)
                if len(combined) >= k:
                    break
            if len(combined) >= k:
                break

        return combined

    def _retrieve_global_knowledge(
        self, query: str
    ) -> Dict[str, List[Document]]:
        decision = self._route_query(query)
        knowledge = {}

        label_map = {
            "01_GTM_Frameworks"               : "GTM Frameworks",
            "02_AI_Business_Solutions"        : "AI Business Solutions",
            "03_Market_Knowledge"             : "Market Knowledge",
            "04_Localization"                 : "Localization",
            "05_Personas"                     : "Personas",
            "06_Technology_and_Digital_Capabilities": "Technology",
            "07_Competitor_Knowledge"         : "Competitors",
        }

        for topic in decision.domains:
            docs = self._retrieve_from_topic(query, topic)
            if docs:
                knowledge[label_map.get(topic, topic)] = docs

        return knowledge

    def _get_client_db(self, client_name: str) -> Optional[Chroma]:
        """Lazy-load and cache the Chroma DB for a client."""
        if client_name in self.client_dbs:
            return self.client_dbs[client_name]

        db_path = os.path.join(
            BASE_DIR, "vector_db", "clients", f"{client_name}_db"
        )
        if not os.path.exists(db_path):
            print(f"No vector DB found for client: {client_name}")
            return None

        client_db = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings
        )
        self.client_dbs[client_name] = client_db

        chunk_count = client_db._collection.count()
        print(f"Loaded client DB [{client_name}] - {chunk_count} chunks.")
        return client_db

    def _retrieve_client_knowledge(
        self, query: str, client_name: str
    ) -> List[Document]:
        client_db = self._get_client_db(client_name)
        if client_db is None:
            return []

        try:
            docs = client_db.max_marginal_relevance_search(
                query=query,
                k=CLIENT_K,
                fetch_k=15
            )
            for doc in docs:
                doc.metadata["retrieval_type"] = "semantic"
            return docs
        except Exception as e:
            print(f"Client retrieval error [{client_name}]: {e}")
            return []

    # ------------------------------------------------------------------
    # Context formatting  (Strategy Justifier: label every snippet)
    # ------------------------------------------------------------------

    def _format_context(
        self,
        global_knowledge: Dict[str, List[Document]],
        client_docs: Optional[List[Document]] = None
    ) -> str:
        """
        Build the context string. Each snippet is prefixed with a source label
        so the model can cite the exact file + page inline:
            [Source: Market Knowledge | saudi-consumer-trends.pdf | p.10]
        """
        sections = []

        for label, docs in global_knowledge.items():
            blocks = []
            for doc in docs:
                meta = doc.metadata or {}
                fname = meta.get("filename", "Unknown")
                page = int(meta.get("page", 0)) + 1
                tag = f"[Source: {label} | {fname} | p.{page}]"
                blocks.append(f"{tag}\n{doc.page_content[:2000]}")
            sections.append(f"## {label}\n" + "\n---\n".join(blocks))

        if client_docs:
            blocks = []
            for doc in client_docs:
                meta = doc.metadata or {}
                fname = meta.get("filename", "Unknown")
                page = int(meta.get("page", 0)) + 1
                tag = f"[Source: Client Knowledge | {fname} | p.{page}]"
                blocks.append(f"{tag}\n{doc.page_content[:2000]}")
            sections.append("## Client Knowledge\n" + "\n---\n".join(blocks))

        return "\n\n".join(sections) if sections else "No relevant context retrieved."

    def _build_sources(
        self,
        global_knowledge: Dict[str, List[Document]],
        client_docs: Optional[List[Document]]
    ) -> List[Dict]:
        sources = []

        for label, docs in global_knowledge.items():
            for doc in docs:
                meta = doc.metadata
                sources.append({
                    "topic"   : label,
                    "filename": meta.get("filename", "Unknown"),
                    "page"    : int(meta.get("page", 0)) + 1,
                })

        if client_docs:
            for doc in client_docs:
                meta = doc.metadata
                sources.append({
                    "topic"   : "Client KB",
                    "filename": meta.get("filename", "Unknown"),
                    "page"    : int(meta.get("page", 0)) + 1,
                })

        return sources

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_gtm_strategy(self, inputs: Dict[str, Any]) -> Dict:
        query            = self._build_strategy_query(inputs)
        refinement       = (inputs.get("refinement") or "").strip()
        goal             = inputs["business_goal"]
        if refinement:
            goal = f"{goal}\n\nIMPORTANT — the user requested this refinement; apply it throughout the strategy: {refinement}"
        global_knowledge = self._retrieve_global_knowledge(query)
        client_docs: List[Document] = []

        if inputs.get("mode") == "Client Mode" and inputs.get("client_name"):
            client_docs = self._retrieve_client_knowledge(
                query, inputs["client_name"]
            )
            print(f"CLIENT DOCS FOUND: {len(client_docs)}")

        context  = self._format_context(global_knowledge, client_docs)
        chain    = GTM_PROMPT | self.llm | StrOutputParser()
        strategy = chain.invoke({
            "product_name"       : inputs["product_name"],
            "product_description": inputs["product_description"],
            "industry"           : inputs["industry"],
            "region"             : inputs["region"],
            "business_goal"      : goal,
            "brand_tone"         : inputs["brand_tone"],
            "mode"               : inputs["mode"],
            "context"            : context,
        })

        return {
            "type"    : "strategy",
            "strategy": strategy,
            "sources" : self._build_sources(global_knowledge, client_docs),
        }

    gap_topics = [
            "07_Competitor_Knowledge",
            "02_AI_Business_Solutions",
            "06_Technology_and_Digital_Capabilities",
            "03_Market_Knowledge",
            "04_Localization",
        ]

    def generate_gtm_options(self, inputs: Dict[str, Any]) -> Dict:
        query            = self._build_strategy_query(inputs)
        global_knowledge = self._retrieve_global_knowledge(query)
        client_docs: List[Document] = []

        if inputs.get("mode") == "Client Mode" and inputs.get("client_name"):
            client_docs = self._retrieve_client_knowledge(
                query, inputs["client_name"]
            )

        context = self._format_context(global_knowledge, client_docs)
        chain   = OPTIONS_PROMPT | self.llm | StrOutputParser()
        options = chain.invoke({
            "product_name"       : inputs["product_name"],
            "product_description": inputs["product_description"],
            "industry"           : inputs["industry"],
            "region"             : inputs["region"],
            "business_goal"      : inputs["business_goal"],
            "context"            : context,
        })

        return {
            "type"   : "options",
            "options": options,
            "sources": self._build_sources(global_knowledge, client_docs),
        }
    
    def simulate_what_if(self, inputs: Dict[str, Any]) -> Dict:

        query = (
            f"What-if GTM scenario for {inputs['product_name']} "
            f"{inputs['product_description']} "
            f"{inputs['industry']} {inputs['region']} "
            f"{inputs['business_goal']} "
            f"Scenario: {inputs['scenario']}"
        )

        global_knowledge = self._retrieve_global_knowledge(query)
        client_docs: List[Document] = []

        if inputs.get("mode") == "Client Mode" and inputs.get("client_name"):
            client_docs = self._retrieve_client_knowledge(
                query,
                inputs["client_name"]
            )

        context = self._format_context(global_knowledge, client_docs)

        chain = WHAT_IF_PROMPT | self.llm | StrOutputParser()

        simulation = chain.invoke({
            "product_name": inputs["product_name"],
            "product_description": inputs["product_description"],
            "industry": inputs["industry"],
            "region": inputs["region"],
            "business_goal": inputs["business_goal"],
            "brand_tone": inputs["brand_tone"],
            "scenario": inputs["scenario"],
            "current_strategy": inputs.get("current_strategy", ""),
            "context": context,
        })

        return {
            "type": "what_if",
            "simulation": simulation,
            "sources": self._build_sources(global_knowledge, client_docs),
        }
    def calculate_confidence_score(self, inputs: Dict[str, Any]) -> Dict:

        query = (
            f"Evaluate GTM strategy confidence for {inputs['product_name']} "
            f"{inputs['product_description']} "
            f"{inputs['industry']} {inputs['region']} "
            f"{inputs['business_goal']} "
            f"Strategy: {inputs['strategy']}"
        )

        global_knowledge = self._retrieve_global_knowledge(query)
        client_docs: List[Document] = []

        if inputs.get("mode") == "Client Mode" and inputs.get("client_name"):
            client_docs = self._retrieve_client_knowledge(
                query,
                inputs["client_name"]
            )

        context = self._format_context(global_knowledge, client_docs)

        chain = CONFIDENCE_SCORE_PROMPT | self.llm | StrOutputParser()

        confidence_report = chain.invoke({
            "product_name": inputs["product_name"],
            "product_description": inputs["product_description"],
            "industry": inputs["industry"],
            "region": inputs["region"],
            "business_goal": inputs["business_goal"],
            "brand_tone": inputs["brand_tone"],
            "strategy": inputs["strategy"],
            "context": context,
        })

        return {
            "type": "confidence_score",
            "confidence_report": confidence_report,
            "sources": self._build_sources(global_knowledge, client_docs),
        }

    def generate_launch_timeline(self, inputs: Dict[str, Any]) -> Dict:

        query = (
            f"Create 30 60 90 day GTM launch timeline for {inputs['product_name']} "
            f"{inputs['product_description']} "
            f"{inputs['industry']} {inputs['region']} "
            f"{inputs['business_goal']}"
        )

        global_knowledge = self._retrieve_global_knowledge(query)
        client_docs: List[Document] = []

        if inputs.get("mode") == "Client Mode" and inputs.get("client_name"):
            client_docs = self._retrieve_client_knowledge(
                query,
                inputs["client_name"]
            )

        context = self._format_context(global_knowledge, client_docs)

        chain = LAUNCH_TIMELINE_PROMPT | self.llm | StrOutputParser()

        timeline = chain.invoke({
            "product_name": inputs["product_name"],
            "product_description": inputs["product_description"],
            "industry": inputs["industry"],
            "region": inputs["region"],
            "business_goal": inputs["business_goal"],
            "brand_tone": inputs["brand_tone"],
            "context": context,
        })

        return {
            "type": "launch_timeline",
            "timeline": timeline,
            "sources": self._build_sources(global_knowledge, client_docs),
        }

    def find_competitive_gaps(self, inputs: Dict[str, Any]) -> Dict:

        query = (
            f"competitors, market landscape, and competitive positioning for "
            f"{inputs['product_name']} - {inputs['product_description']} "
            f"in the {inputs['industry']} industry, {inputs['region']} market"
        )

        global_knowledge = self._retrieve_global_knowledge(query)
        client_docs: List[Document] = []

        if inputs.get("mode") == "Client Mode" and inputs.get("client_name"):
            client_docs = self._retrieve_client_knowledge(
                query, inputs["client_name"]
            )

        context = self._format_context(global_knowledge, client_docs)
        chain   = GAP_PROMPT | self.llm | StrOutputParser()

        gaps = chain.invoke({
            "product_name"       : inputs["product_name"],
            "product_description": inputs["product_description"],
            "industry"           : inputs["industry"],
            "region"             : inputs["region"],
            "business_goal"      : inputs["business_goal"],
            "client_name"        : inputs.get("client_name", ""),
            "context"            : context,
        })

        return {
            "type"   : "gaps",
            "gaps"   : gaps,
            "sources": self._build_sources(global_knowledge, client_docs),
        }

    def score_brand_match(self, strategy: str, brand_tone: str) -> Dict:
        chain      = BRAND_SCORE_PROMPT | self.llm | StrOutputParser()
        assessment = chain.invoke({
            "strategy"  : strategy,
            "brand_tone": brand_tone,
        })
        return {
            "type"      : "brand_score",
            "assessment": assessment,
        }

    def ask_aura(
        self, query: str, client_name: Optional[str] = None
    ) -> Dict:

        expanded_query = (
            f"{query} "
            f"B2B enterprise leads Saudi Arabia AI consulting data analytics automation "
            f"decision makers CIO CTO CDO LinkedIn account-based marketing partnerships "
            f"industry events webinars case studies enterprise trust"
        )

        label_map = {
            "01_GTM_Frameworks": "GTM Frameworks",
            "02_AI_Business_Solutions": "AI Business Solutions",
            "03_Market_Knowledge": "Market Knowledge",
            "04_Localization": "Localization",
            "05_Personas": "Personas",
            "07_Competitor_Knowledge": "Competitors",
        }

        ask_topics = [
            "01_GTM_Frameworks",
            "02_AI_Business_Solutions",
            "03_Market_Knowledge",
            "04_Localization",
            "05_Personas",
            "07_Competitor_Knowledge",
        ]

        global_knowledge = {}

        for topic in ask_topics:
            docs = self._retrieve_from_topic(expanded_query, topic, k=2)
            if docs:
                global_knowledge[label_map.get(topic, topic)] = docs

        client_docs: List[Document] = []

        if client_name:
            client_docs = self._retrieve_client_knowledge(
                expanded_query,
                client_name
            )

        context = self._format_context(global_knowledge, client_docs)

        chain = ASK_AURA_PROMPT | self.llm | StrOutputParser()

        answer = chain.invoke({
            "query": query,
            "context": context,
            "client_name": client_name or "No client selected"
        })

        return {
            "type": "answer",
            "answer": answer,
            "sources": self._build_sources(global_knowledge, client_docs),
        }

    def run(self, payload: Dict[str, Any]) -> Dict:
        if payload.get("query"):
            return self.ask_aura(
                query=payload["query"],
                client_name=payload.get("client_name")
            )
        return self.generate_gtm_strategy(payload)


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

GTM_PROMPT = ChatPromptTemplate.from_template("""
You are AuraGTM, a senior Go-To-Market strategist.

Your task is to generate a practical, executive-level GTM strategy grounded primarily in the retrieved knowledge. When Client Knowledge is available, prioritize it over general knowledge and build a strategy specific to that client.

If some information is unavailable, use your strategic expertise and explicitly state assumptions.

PRODUCT INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Product Name:
{product_name}

Product Description:
{product_description}

Industry:
{industry}

Region:
{region}

Business Goal:
{business_goal}

Brand Tone:
{brand_tone}

Mode:
{mode}

RETRIEVED CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{context}

CLIENT KNOWLEDGE PRIORITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When Client Knowledge is available:

1. Treat Client Knowledge as the PRIMARY source of truth.

2. Use Global Knowledge only to enrich and support recommendations.

3. If there is any conflict between Client Knowledge and Global Knowledge, always prioritize Client Knowledge.

4. Tailor all recommendations, personas, messaging, competitor analysis, marketing channels, GTM roadmap, and content ideas to the client's business, services, capabilities, projects, expertise, and market positioning.

5. Avoid generating generic strategies when Client Knowledge is available.

6. Explicitly use information from Client Knowledge whenever relevant.

7. Reference the client's services, solutions, technologies, industries, and past projects whenever possible.

8. Build recommendations that realistically align with the client's actual capabilities rather than generic market assumptions.

GROUNDING & JUSTIFICATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Every snippet in the RETRIEVED CONTEXT is labeled with its source, like:
[Source: Market Knowledge | saudi-consumer-trends.pdf | p.10]

Follow these rules throughout the strategy:

1. Base your recommendations, insights, and claims on the RETRIEVED CONTEXT whenever it is relevant. Let the evidence in the context shape what you write.

2. Do NOT print source labels, filenames, page numbers, or any inline citations in the strategy text. Write clean, professional prose with no bracketed references of any kind.

3. Briefly justify your major recommendations with a short "because ..." so the reasoning is explicit to the reader — explain the logic, without naming any source files.

4. Never state facts that contradict the retrieved context, and never fabricate specific figures or data points that the context does not support.

INSTRUCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Generate EXACTLY the following sections:

## 1. Target Audience

Primary and secondary audiences.

Include:
- demographics
- psychographics
- behavioral traits

---

## 2. Buyer Personas

Create 2–3 personas.

For each include:

- Name
- Role
- Goals
- Pain Points
- Preferred Channels

---

## 3. Localization Insights

Include:

- cultural considerations
- language preferences
- local dynamics
- regulatory considerations

---

## 4. Competitor Analysis

For each competitor:

- strengths
- weaknesses
- positioning
- differentiation opportunities

---

## 5. Positioning & Messaging

Provide:

- positioning statement
- 3 key messages
- suggested tagline

---

## 6. Marketing Channels

Top 5 channels.

For each include:

- rationale
- content type
- priority

---

## 7. GTM Roadmap

Phase 1:
Launch Preparation

Phase 2:
Market Entry

Phase 3:
Scale & Optimization

Include:

- actions
- KPIs
- milestones

---

## 8. Content Ideas

Generate 8 ideas.

Format:

[Content Type] — Title — Platform — Goal

---

## 9. Assumptions & Risks

List major assumptions.

Highlight execution risks.

---

## 10. Executive Summary

Provide concise recommendations.

Ground recommendations in the retrieved context whenever possible. Do not print filenames, page numbers, or inline citations anywhere in the strategy.
""")

ASK_AURA_PROMPT = ChatPromptTemplate.from_template("""
You are AuraGTM, a senior B2B go-to-market strategist.

Answer the user's strategic question clearly and professionally.

CLIENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{client_name}

CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context}

QUESTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{query}

RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use retrieved context first.
If Client Knowledge is available, prioritize it.
Focus on B2B GTM strategy, enterprise buyers, decision-makers, trust-building, lead generation, and market entry.
For enterprise lead generation, prioritize channels such as LinkedIn, account-based outreach, webinars, industry events, partnerships, case studies, whitepapers, executive briefings, and referral networks.
Do NOT recommend B2C channels like OTT platforms, entertainment ads, influencer campaigns, or mass consumer channels unless the question clearly asks for B2C.
Do NOT print source labels, filenames, page numbers, or inline citations.
Never invent exact numbers or unsupported facts.

Provide:

1. Direct Answer

2. Strategic Implications

3. Recommended Actions

4. Risks or Considerations
""")


GAP_PROMPT = ChatPromptTemplate.from_template("""
You are AuraGTM, a senior GTM strategist.

Your task is to identify competitive gaps and opportunities for the product below.

PRODUCT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name: {product_name}
Description: {product_description}
Industry: {industry}
Region: {region}
Goal: {business_goal}
Client Name: {client_name}

RETRIEVED CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context}

RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Prioritize Client Knowledge if available.
Focus on the product, industry, region, competitors, positioning, and differentiation.
Do NOT over-focus on unrelated industries from the retrieved context.
If a retrieved source is broad, use only the parts that are relevant to this product.
Do NOT invent exact numbers or competitor facts if they are not in the context.
Do NOT print filenames, page numbers, or inline citations.

Output EXACTLY these sections:

## Market & Competitor Landscape
Explain the relevant competitive context for this product and region.

## Your Strengths
- List 4-6 strengths based on the product and client context.

## Gaps & Vulnerabilities
- List 4-6 realistic weaknesses, missing assets, or risks.

## Opportunities to Win
- List 4-6 practical opportunities to differentiate.

## Priority Moves
- List the top 3-5 actions the company should take next.
""")


OPTIONS_PROMPT = ChatPromptTemplate.from_template("""
You are AuraGTM. Propose THREE clearly different go-to-market directions for the product below.

PRODUCT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name: {product_name}
Description: {product_description}
Industry: {industry}
Region: {region}
Goal: {business_goal}

RETRIEVED CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context}

RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use the retrieved context where relevant.
Do NOT print source labels, filenames, page numbers, or inline citations.
The three options must be clearly different.
You MUST include a Recommended Option section at the end.
The recommended choice must be exactly one of: Option A, Option B, or Option C.

Output EXACTLY this structure:

## Option A — <short name>
- Best for: <who this suits>
- Positioning: <one line>
- Primary channels: <2-3 channels>
- Why it could win: <one line>
- Main risk: <one line>

## Option B — <short name>
- Best for: <who this suits>
- Positioning: <one line>
- Primary channels: <2-3 channels>
- Why it could win: <one line>
- Main risk: <one line>

## Option C — <short name>
- Best for: <who this suits>
- Positioning: <one line>
- Primary channels: <2-3 channels>
- Why it could win: <one line>
- Main risk: <one line>

## Recommended Option
- Choice: Option A / Option B / Option C
- Reason: <why this is the strongest option>
- Confidence: High / Medium / Low
""")

WHAT_IF_PROMPT = ChatPromptTemplate.from_template("""
You are AuraGTM, a senior GTM strategy simulator.

Your task is to analyze how the GTM strategy should change under a specific what-if scenario.

PRODUCT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name: {product_name}
Description: {product_description}
Industry: {industry}
Region: {region}
Goal: {business_goal}
Brand Tone: {brand_tone}

WHAT-IF SCENARIO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{scenario}

CURRENT STRATEGY IF PROVIDED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{current_strategy}

RETRIEVED CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context}

RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use retrieved context when relevant.
Do NOT print source labels, filenames, page numbers, or inline citations.
Do not invent exact numbers if they are not provided.
Be practical and decision-oriented.

Output EXACTLY these sections:

## Scenario Impact
Explain how this scenario changes the GTM plan in 3-5 sentences.

## What Should Change
- List 4-6 concrete changes to audience, positioning, channels, messaging, pricing, or launch plan.

## What Should Stay the Same
- List 2-4 parts of the original GTM direction that should remain stable.

## New Risks
- List 3-5 risks created by this scenario.

## Recommended Response
Give one clear recommendation for how the team should adjust the GTM strategy.

## Confidence
High / Medium / Low, with one short reason.
""")

CONFIDENCE_SCORE_PROMPT = ChatPromptTemplate.from_template("""
You are AuraGTM, a GTM quality evaluator.

Your task is to evaluate how reliable and complete the GTM strategy is based on:
1. Product clarity
2. Market and region fit
3. Retrieved context strength
4. Client knowledge availability
5. Practicality of the strategy
6. Missing information or assumptions

PRODUCT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name: {product_name}
Description: {product_description}
Industry: {industry}
Region: {region}
Goal: {business_goal}
Brand Tone: {brand_tone}

STRATEGY TO EVALUATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{strategy}

RETRIEVED CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context}

RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Do NOT print source labels, filenames, page numbers, or inline citations.
Do NOT give a perfect score unless the strategy is clearly supported and complete.
If important inputs are missing, reduce the score.
Be strict but fair.
Do not invent exact market numbers if they are not provided.

Output EXACTLY this structure:

## Overall Confidence Score
Score: <number from 0 to 100>/100
Level: High / Medium / Low

## Why This Score
Explain the score in 3-5 sentences.

## Strong Evidence
- List 3-5 points that make the strategy stronger.

## Weaknesses / Missing Inputs
- List 3-5 missing details, weak areas, or assumptions.

## Improvement Suggestions
- List 3-5 actions that would increase the confidence score.

## Final Assessment
Give one short final judgment on whether this strategy is ready for use, needs refinement, or needs more client data.
""")

LAUNCH_TIMELINE_PROMPT = ChatPromptTemplate.from_template("""
You are AuraGTM, a GTM launch planning expert.

Your task is to create a practical 30/60/90-day launch timeline for the product below.

PRODUCT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name: {product_name}
Description: {product_description}
Industry: {industry}
Region: {region}
Goal: {business_goal}
Brand Tone: {brand_tone}

RETRIEVED CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context}

RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use retrieved context when relevant.
Do NOT print source labels, filenames, page numbers, or inline citations.
Make the plan realistic and execution-focused.
Do not invent budgets, dates, or exact market numbers if not provided.
Focus on actions, ownership, channels, and success metrics.

Output EXACTLY this structure:

## 30 Days — Foundation
### Main Objective
Explain the goal for the first 30 days.

### Key Actions
- List 5-7 practical actions.

### Recommended Channels
- List the best channels for this phase.

### Success Metrics
- List 3-5 measurable indicators.

## 60 Days — Activation
### Main Objective
Explain the goal for days 31-60.

### Key Actions
- List 5-7 practical actions.

### Recommended Channels
- List the best channels for this phase.

### Success Metrics
- List 3-5 measurable indicators.

## 90 Days — Scale
### Main Objective
Explain the goal for days 61-90.

### Key Actions
- List 5-7 practical actions.

### Recommended Channels
- List the best channels for this phase.

### Success Metrics
- List 3-5 measurable indicators.

## Priority Owner Roles
- List the roles needed to execute the plan.

## Final Launch Recommendation
Give one clear recommendation for how the team should execute this launch timeline.
""")

BRAND_SCORE_PROMPT = ChatPromptTemplate.from_template("""
You are AuraGTM, evaluating how well a generated GTM strategy matches the intended brand tone.

INTENDED BRAND TONE: {brand_tone}

STRATEGY TO EVALUATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{strategy}

Assess how well the strategy's language and recommendations fit the intended tone. Output EXACTLY this, with headings on their own line starting with "## ":

## Brand Match: <NN>/100
(Choose a realistic number from 0 to 100.)

## What fits
- 2-3 short points on where the strategy matches the intended tone.

## What to adjust
- 2-3 short, concrete suggestions to bring it closer to the intended tone.

Do NOT print filenames or citations. Keep it concise.
""")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_engine: Optional[AuraGTMEngine] = None


def get_engine() -> AuraGTMEngine:
    global _engine
    if _engine is None:
        _engine = AuraGTMEngine()
    return _engine