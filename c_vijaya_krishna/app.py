import os
from PyPDF2 import PdfReader
import streamlit as st
from dotenv import load_dotenv
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from reader import Reader

image_width = 150
image_height = 150
image = r"D:\\Projects\\DocumentGPT\\Converso\\logo.png"
#Sidebar contents
with st.sidebar:
    # st.image(image, width=image_width, caption="Your Image Caption")
    st.title('Converso')
    add_vertical_space(5)
    st.write('Made by vijay')


def main():
    st.header("Document Assistant")
    load_dotenv()
    #upload File
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    if pdf is not None:

        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        
        if len(text)==0:
            st.write(pdf.name)
            with open(os.path.join("tempDir",pdf.name),"wb") as f:
                f.write(pdf.getvalue())
            text=Reader().read(os.path.join("tempDir",pdf.name)).get_output()
            

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        
        #embeddings
        store_name = pdf.name[:-4]
        
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectoreStore = pickle.load(f)
            st.write('Embeddings Loaded from the disk')
        else:
            embeddings = OpenAIEmbeddings()
            VectoreStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectoreStore, f)
            st.write("Embedding Computation Completed")

        #Accept user query
        query = st.text_input("Ask questions about your PDF file: ")

        if query:
            docs = VectoreStore.similarity_search(query=query, k=3)
            llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.5)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.write(response)

            #st.write(docs)

        # st.write(text)
        

if __name__=='__main__':
    main()