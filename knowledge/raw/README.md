# RAG Knowledge Sources

This directory is used for the raw reference documents of the manufacturing RAG pipeline.

The original PDF files are not included in this repository.

## Sources

1. UCI Machine Learning Repository
   - AI4I 2020 Predictive Maintenance Dataset documentation

2. NASA
   - Reliability-Centered Maintenance Guide
   - September 2008

3. NIST
   - NIST AMS 400-1
   - August 2019

4. U.S. Department of Energy
   - Operations & Maintenance Best Practices
   - Release 3.0
   - August 2010

5. OSHA
   - Lockout/Tagout Fact Sheet
   - OSHA FS-3529
   - December 2022

## Usage

Place the corresponding PDF files in this directory before running the RAG ingestion pipeline.

The ingestion pipeline:

1. loads the PDF documents,
2. performs section-aware chunking,
3. generates BGE-M3 embeddings,
4. stores vectors in Qdrant,
5. retrieves dense candidates,
6. reranks them using BGE Reranker v2 M3.

Local Qdrant data is also excluded from Git.