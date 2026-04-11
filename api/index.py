# -*- coding: utf-8 -*-
import os
import re
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# =====================================
# 1. FastAPI + CORS
# =====================================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================
# 2. تحديد المسارات (مهم جداً للـ Vercel)
# =====================================
# هذا السطر يحدد مكان المجلد الرئيسي للمشروع
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")

# =====================================
# 3. تحميل قاعدة البيانات (Vectorstore)
# =====================================
emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# نحاول تحميل قاعدة البيانات، وإذا لم نجدها لا يتوقف السيرفر
try:
    vectorstore = FAISS.load_local(INDEX_PATH, emb, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    print("✅ Vectorstore loaded successfully from:", INDEX_PATH)
except Exception as e:
    vectorstore = None
    retriever = None
    print(f"❌ Error loading Vectorstore: {e}")

# =====================================
# 4. نموذج Groq
# =====================================
# نستخدم os.getenv لجلب المفتاح من إعدادات Vercel (أمان أعلى)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=1500
)

# =====================================
# 5. الوظائف المساعدة (الترجمة وكشف اللغة)
# =====================================
def is_arabic(text):
    return bool(re.search(r'[\u0600-\u06FF]', text))

async def translate_to_english(text):
    if not is_arabic(text):
        return text
    messages = [
        ("system", "Translate the following medical terms from Arabic to English only."),
        ("human", text),
    ]
    try:
        response = llm.invoke(messages)
        return response.content.strip()
    except:
        return text

# =====================================
# 6. Prompt & Chain
# =====================================
prompt = ChatPromptTemplate.from_template(
    """You are a detailed medical assistant. Answer in **English only**.
Context:
{context}
Question: {question}

**Detailed Answer:**
- [Definition]
- [Function]
- [Examples]
- [Importance]
- [Summary]

If no info, say: "No relevant information found in the documents."
"""
)

def format_context(docs):
    return "\n\n".join(
        f"[From: {d.metadata.get('source','?')} - {('Slide' if d.metadata.get('extension','').lower() in ['.ppt','.pptx'] else 'Page')} {d.metadata.get('page','?')} {d.metadata.get('extension','').upper()}]\n{d.page_content}"
        for d in docs
    )

# بناء السلسلة (Chain)
if retriever:
    chain = (
        {"context": RunnableLambda(lambda x: retriever.invoke(x["question"])),
         "question": RunnablePassthrough()}
        | RunnablePassthrough.assign(context=lambda x: format_context(x["context"]))
        | prompt
        | llm
        | StrOutputParser()
    )
else:
    chain = None

# =====================================
# 7. API Routes
# =====================================
@app.get("/")
async def root():
    return {"message": "Medical Chatbot API is running!"}

@app.get("/api/chat")
async def chat(query: str = Query(...)):
    if not chain:
        return {"answer": "System not ready: Vectorstore not found.", "source": "N/A"}

    try:
        english_query = await translate_to_english(query)
        docs = retriever.invoke(english_query)
        
        if not docs:
            return {"answer": "No relevant information found in the documents.", "source": "N/A"}

        answer = chain.invoke({"question": english_query})
        
        sources = []
        for d in docs:
            src = d.metadata.get("source", "?")
            page = d.metadata.get("page", "?")
            ext = d.metadata.get("extension", "").upper()
            unit = "Slide" if ext in ["PPT", "PPTX"] else "Page"
            sources.append(f"{src} ({unit} {page})")

        unique_sources = list(dict.fromkeys(sources))
        final_answer = f"{answer.strip()}\n\n**Sources:**\n" + "\n".join([f"• {s}" for s in unique_sources])

        return {"answer": final_answer, "source": ", ".join(unique_sources)}

    except Exception as e:
        return {"answer": f"Error: {str(e)}", "source": "N/A"}