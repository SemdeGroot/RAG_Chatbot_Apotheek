---
title: Apotheek Chatbot
emoji: 💊
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Apotheek Chatbot (RAG)

Een lichte **Flask** webapp die met **FAISS** + **Sentence Transformers** relevante passages uit enkele *apotheek.nl* teksten (lokaal geparseerd) ophaalt en via **Groq LLM** een beknopt NL-antwoord geeft. Ontworpen als Proof of Concept.