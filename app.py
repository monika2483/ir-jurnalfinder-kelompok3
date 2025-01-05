import streamlit as st
from model import JournalSearchModel

st.title("Jurnal Finder")
st.write("Temukan jurnal yang relevan dengan mudah.")

file_path = "project IR\dataset IR.xlsx"
stopword_path = "stopword-list.txt"
kata_dasar_path = "kata-dasar.txt"
model = JournalSearchModel(file_path, stopword_path, kata_dasar_path)

query = st.text_input("Cari Jurnal", placeholder="Masukkan kata kunci")

if st.button("Cari"):
    if query:
        query_tokens, query_vector = model.process_query(query)
        matched_docs = model.retrieve_by_inverted_index(query_tokens)

        if matched_docs:
            ranked_docs = model.rank_documents(query_vector, matched_docs)
            st.write("Dokumen yang ditemukan :")
            for doc_id, _ in ranked_docs:

                title = model.titles[doc_id]
                author = model.dataset["Penulis"][doc_id]
                year = model.dataset["Tahun Terbit"][doc_id]
                abstract = model.dataset["Abstrak"][doc_id]
                url = model.dataset["Link Jurnal"][doc_id]

                st.subheader(title)
                st.write(f"**Penulis**: {author}")
                st.write(f"**Tahun Terbit**: {year}")
                st.write(f"**Preview Abstrak**: {abstract[:300]}...")
                st.write(f"[Baca Jurnal Lengkap]( {url} )")
                st.markdown("---")
        else:
            st.write("Tidak ada dokumen yang ditemukan.")
    else:
        st.write("Harap masukkan kata kunci terlebih dahulu.")
