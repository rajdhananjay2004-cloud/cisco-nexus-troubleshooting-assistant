import html
import logging
import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_DIR = Path(__file__).resolve().parent
PDF_PATH = PROJECT_DIR / "data" / "cisco_nexus_9000_troubleshooting_guide.pdf"
CHROMA_PATH = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "cisco_nexus_troubleshooting"
MODEL_NAME = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVAL_K = 5
SYSTEM_PROMPT = """You are a technical assistant for the Cisco Nexus 9000 Series Troubleshooting Guide.

Use only the retrieved context as evidence. Do not use general networking
knowledge to fill gaps or invent technical facts, commands, or procedures.
Answer direct questions concisely. For broader questions, synthesize relevant
passages into clear, numbered steps when several actions are available, without
claiming unsupported details. A heading or table-of-contents entry alone does
not support a procedure. Summarize process details in your own words instead of
listing copied fragments; do not present headings or references as actions.
Ignore retrieved snippets that contain only section titles, contents entries,
or additional-reference listings. For example, "Using Diagnostics" alone is
not an instruction to run diagnostics.
For a general recurring issue, omit upgrade-recovery steps unless the user asks
about a failed upgrade.
If the context does not contain enough information, say so explicitly. Cite
supporting pages using the exact [Page N] labels shown in the context, and do
not cite pages that were not provided."""

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
load_dotenv(PROJECT_DIR / ".env")

retriever = None
llm = None
startup_message = ""


def build_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_K},
    )


def initialize_rag():
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is missing from the project .env file.")
    if not PDF_PATH.is_file():
        raise FileNotFoundError(
            f"PDF not found. Place the Cisco guide at: {PDF_PATH}"
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PATH),
    )
    if vectorstore._collection.count() == 0:
        pages = PyMuPDFLoader(str(PDF_PATH)).load()
        if not any(page.page_content.strip() for page in pages):
            raise ValueError(
                "The PDF contains no extractable text. Scanned PDFs need OCR, "
                "which this demo does not include."
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=60,
            add_start_index=True,
        )
        chunks = splitter.split_documents(pages)
        if not chunks:
            raise ValueError("No searchable text chunks could be created from the PDF.")
        vectorstore.add_documents(chunks)
        logger.info("Indexed %s text chunks from the PDF.", len(chunks))
    else:
        logger.info("Reusing the existing ChromaDB index.")

    active_retriever = build_retriever(vectorstore)
    active_llm = ChatGroq(
        model=MODEL_NAME,
        temperature=0.1,
        api_key=os.environ["GROQ_API_KEY"],
    )
    return active_retriever, active_llm


def answer_question(question, active_retriever=None, active_llm=None):
    if not question or not question.strip():
        return "Please enter a question."
    if active_retriever is None or active_llm is None:
        raise RuntimeError("The assistant is not initialized.")

    documents = active_retriever.invoke(question.strip())[:RETRIEVAL_K]
    if not documents or not any(document.page_content.strip() for document in documents):
        return (
            "The supplied troubleshooting guide context does not provide "
            "sufficient information to answer this question."
        )

    context_parts = []
    for document in documents:
        page = document.metadata.get("page")
        if isinstance(page, int):
            page_number = page + 1
            page_label = f"Page {page_number}"
        else:
            page_label = "Page unavailable"
        context_parts.append(f"[{page_label}]\n{document.page_content}")

    context = "\n\n".join(context_parts)
    user_prompt = f"""Retrieved document context:
{context}

User question:
{question.strip()}"""

    from langchain_core.messages import HumanMessage, SystemMessage

    result = active_llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"{user_prompt}\n\n"
                    "Return only short, exact source lines that answer the question."
                )
            ),
        ]
    )
    generated_answer = result.content
    if not isinstance(generated_answer, str):
        generated_answer = str(generated_answer)
    generated_answer = generated_answer.strip()
    if not generated_answer:
        return "The model returned an empty answer. Please try asking the question again."

    question_lower = question.casefold()
    irrelevant_terms = []
    if "upgrade" not in question_lower:
        irrelevant_terms.append("upgrade")
    if "diagnostic" not in question_lower:
        irrelevant_terms.append("diagnostic")
    generated_answer = "\n".join(
        line
        for line in generated_answer.splitlines()
        if not any(term in line.casefold() for term in irrelevant_terms)
    ).strip()

    return generated_answer or (
        "The retrieved context does not provide relevant information to answer "
        "this question."
    )


def render_history(entries):
    if not entries:
        return "Your questions and answers will appear here."
    sections = []
    for number, (question, answer) in enumerate(entries, start=1):
        sections.append(
            f"### {number}. {html.escape(question)}\n\n{html.escape(answer)}"
        )
    return "\n\n---\n\n".join(sections)


def submit_question(question, chat_history, session_history):
    chat_history = list(chat_history or [])
    session_history = list(session_history or [])
    if not question or not question.strip():
        return "", chat_history, session_history, render_history(session_history)
    if retriever is None or llm is None:
        response = startup_message or "The assistant is not ready. Check the terminal for details."
    else:
        try:
            response = answer_question(question, retriever, llm)
        except Exception:
            logger.exception("Question failed.")
            response = (
                "I could not complete that request because the document search "
                "or language model failed. Check the terminal and your Groq "
                "configuration, then try again."
            )

    chat_history.extend(
        [
            {"role": "user", "content": question.strip()},
            {"role": "assistant", "content": response},
        ]
    )
    session_history.append((question.strip(), response))
    return "", chat_history, session_history, render_history(session_history)


def clear_history():
    return [], [], "Your questions and answers will appear here."


if __name__ == "__main__":
    try:
        retriever, llm = initialize_rag()
    except Exception as error:
        logger.error(
            "Assistant initialization failed (%s): %s",
            type(error).__name__,
            error,
        )
        startup_message = (
            f"Assistant unavailable: {error} "
            "Check the project setup and terminal output, then restart the app."
        )


with gr.Blocks(title="Cisco Nexus 9000 Troubleshooting Assistant") as demo:
    gr.Markdown("# Cisco Nexus 9000 Troubleshooting Assistant")
    gr.Markdown(
        "Ask questions grounded in the supplied Cisco Nexus 9000 "
        "Series Troubleshooting Guide."
    )
    if startup_message:
        gr.Markdown(f"**Setup issue:** {startup_message}")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Conversation History")
            history_display = gr.Markdown("Your questions and answers will appear here.")
            clear_button = gr.Button("Clear History")
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Chat", type="messages", height=520)
            question_box = gr.Textbox(
                label="Question",
                placeholder="Ask about troubleshooting the Cisco Nexus 9000...",
                lines=2,
            )
            with gr.Row():
                submit_button = gr.Button("Ask", variant="primary")
                reset_chat_button = gr.Button("Clear Chat")

    session_history = gr.State([])
    submit_button.click(
        submit_question,
        inputs=[question_box, chatbot, session_history],
        outputs=[question_box, chatbot, session_history, history_display],
    )
    question_box.submit(
        submit_question,
        inputs=[question_box, chatbot, session_history],
        outputs=[question_box, chatbot, session_history, history_display],
    )
    clear_button.click(
        clear_history,
        outputs=[chatbot, session_history, history_display],
    )
    reset_chat_button.click(
        clear_history,
        outputs=[chatbot, session_history, history_display],
    )


if __name__ == "__main__":
    demo.launch()