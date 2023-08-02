import streamlit as st

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

from dotenv import load_dotenv

    
if __name__ == "__main__":
    load_dotenv()

    embeddings = OpenAIEmbeddings()
    documents_idx = FAISS.load_local("parsed_data_conv.idx", embeddings)

    llm = ChatOpenAI()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=documents_idx.as_retriever(),
        return_source_documents=True,
    )
    
    st.markdown("### Tinkoff QA Bot")
    
    query = st.text_input("Задайте свой вопрос:")
    if query:
        answer = qa_chain(query)
        st.markdown("**Ответ:**")
        st.markdown(answer["result"])
        
        message_href = ["Подробнее:"]
        
        for doc_i, doc in enumerate(answer["source_documents"], 1):
            href = doc.metadata["source"]
            href = "https://tinkoff.ru" + href + "?card=q" + str(doc.metadata["seq_num"])
            message_href.append(f"- [Ссылка]({href})")
            
        st.markdown("\n".join(message_href))
