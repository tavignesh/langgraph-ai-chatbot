from typing import List, Optional, Tuple
from pydantic import BaseModel
from langchain.schema import Document
from langgraph.graph import StateGraph
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.chat_models import ChatOllama
import re

model_local = ChatOllama(model="mistral")

embedding_model = OllamaEmbeddings(model='nomic-embed-text')
docs = [WebBaseLoader(url).load() for url in [
    "https://ollama.com/",
    "https://ollama.com/blog/openai-compatibility"
]]

docs = [item for sublist in docs for item in sublist]
for doc in docs:
    doc.metadata['access'] = 'all'  # For testing

splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
split_docs = splitter.split_documents(docs)

vectorstore = Chroma.from_documents(split_docs, embedding=embedding_model, collection_name="test-rag")

retriever = vectorstore.as_retriever()

aftertemplate = """You are a helpful assistant. Use the conversation history for the context of the conversation and the provided context i needed to answer the user's current question.

Conversation history:
{history}

Relevant context:
{context}

Current question:
{question}
"""
afterprompt = ChatPromptTemplate.from_template(aftertemplate)



class RAGState(BaseModel):
    query: str
    access_level: str
    context_given: bool = False
    retrieved_docs: Optional[List[Document]] = []
    filtered_docs: Optional[List[Document]] = []
    history: List[Tuple[str, str]] = []
    response: Optional[str] = None

def make_retriever_node(retriever):
    def retriever_node(state: RAGState) -> RAGState:
        retrieved_docs = retriever.get_relevant_documents(state.query)
        return RAGState(
            query=state.query,
            access_level=state.access_level,
            retrieved_docs=retrieved_docs,
            filtered_docs=[],
            history=state.history,
            response=None
        )
    return retriever_node


def access_control_node(state: RAGState) -> RAGState:
    access_level = state.access_level.lower()
    allowed_docs = []

    for doc in state.retrieved_docs or []:
        access = doc.metadata.get("access", "all").lower()
        if access_level == "ceo":
            allowed_docs.append(doc)
        elif access_level == "employee" and access in ("employee", "all"):
            allowed_docs.append(doc)
        elif access_level not in ("ceo", "employee") and access == "all":
            allowed_docs.append(doc)

    return RAGState(
        query=state.query,
        access_level=state.access_level,
        retrieved_docs=state.retrieved_docs,
        filtered_docs=allowed_docs,
        history=state.history,
        response=None
    )


after_rag_chain = afterprompt | model_local | StrOutputParser()


def rag_node(state: RAGState) -> RAGState:
    context = "\n\n".join(doc.page_content for doc in state.filtered_docs)

    history_text = ""
    for q, a in state.history:
        history_text += f"User: {q}\nAssistant: {a}\n"

    response = after_rag_chain.invoke({
        "context": context,
        "question": state.query,
        "history": history_text
    })

    return RAGState(
        query=state.query,
        access_level=state.access_level,
        retrieved_docs=state.retrieved_docs,
        filtered_docs=state.filtered_docs,
        history=state.history + [(state.query, response)],
        response=response
    )


def augment_query_node(state: RAGState) -> RAGState:
    augment_template = """You are a query augmentation assistant.

    Original question:
    {question}

    Rephrase or expand the question to improve information retrieval. Be specific, and include any implicit context if needed.

    Improved query:"""

    augment_prompt = ChatPromptTemplate.from_template(augment_template)
    augment_chain = augment_prompt | model_local | StrOutputParser()

    improved_query = augment_chain.invoke({"question": state.query})
    return RAGState(
        query=improved_query,
        access_level=state.access_level,
        retrieved_docs=[],
        filtered_docs=[],
        history=state.history,
        response=None
    )


def user_context_node(state: RAGState) -> RAGState:
    url_match = re.search(r'(https?://[^\s]+)', state.query)
    if url_match:
        url = url_match.group(1)
        try:
            docs = WebBaseLoader(url).load()
            for doc in docs:
                doc.metadata['access'] = 'all'

            return RAGState(
                query=state.query,
                access_level=state.access_level,
                context_given=True,  # Set the flag
                retrieved_docs=docs,
                filtered_docs=docs,
                history=state.history,
                response=None
            )
        except Exception as e:
            print(f"Failed to load user URL: {e}")
    return state



def spelling_correction_node(state: RAGState) -> RAGState:
    spell_template = """You are a spelling correction assistant.

    Original input:
    {query}

    Corrected version (fix only spelling mistakes, keep meaning and tone):"""

    spell_prompt = ChatPromptTemplate.from_template(spell_template)
    spell_chain = spell_prompt | model_local | StrOutputParser()

    corrected_query = spell_chain.invoke({"query": state.query})
    print(corrected_query)

    return RAGState(
        query=corrected_query,
        access_level=state.access_level,
        retrieved_docs=[],
        filtered_docs=[],
        history=state.history,
        response=None
    )


def final_node(state: RAGState) -> RAGState:
    return state



retriever_node = make_retriever_node(retriever)
graph = StateGraph(RAGState)

graph.set_entry_point("user_context_check")

graph.add_node("user_context_check", user_context_node)

graph.add_conditional_edges(
    "user_context_check",
    lambda state: "rag" if state.context_given else "spell_correction"
)

graph.add_node("spell_correction", spelling_correction_node)
graph.add_edge("spell_correction", "augment_query")

graph.add_node("augment_query", augment_query_node)
graph.add_edge("augment_query", "retriever")

graph.add_node("retriever", retriever_node)
graph.add_edge("retriever", "access_control")

graph.add_node("access_control", access_control_node)
graph.add_edge("access_control", "rag")

graph.add_node("rag", rag_node)
graph.add_edge("rag", "final")

graph.add_node("final", final_node)

graph.set_finish_point("final")

app = graph.compile()


state = RAGState(query="", access_level="employee", history=[])

while True:
    user_input = input("User: ")
    if user_input.lower().strip() == "stop":
        break

    state.query = user_input

    result = app.invoke(state)
    state = RAGState(**result)

    print(f"\nAssistant: {state.response}\n")