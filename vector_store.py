import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import pickle
import streamlit as st

class VectorStore:
    """
    Enhanced vector store with multiple search methods and document management.
    Combines sentence transformers with TF-IDF for hybrid search.
    """
    
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        self.faiss_index = None
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.8,
            min_df=2
        )
        self.tfidf_matrix = None
        
        # Document storage
        self.chunks = []
        self.chunk_metadata = []  # Store source file, chunk index, etc.
        self.document_mapping = {}  # Map document names to chunk indices
        
        # Initialize models
        self._load_models()
    
    def _load_models(self):
        """Load embedding model with caching"""
        try:
            if self.embedding_model is None:
                with st.spinner("Loading embedding model..."):
                    self.embedding_model = SentenceTransformer(self.embedding_model_name)
                    st.success("✅ Embedding model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading embedding model: {str(e)}")
            raise e
    
    def add_documents(self, chunks, document_name):
        """Add new documents to the vector store"""
        try:
            if not chunks:
                raise ValueError("No chunks provided")
            
            # Store starting index for this document
            start_index = len(self.chunks)
            
            # Add chunks and metadata
            for i, chunk in enumerate(chunks):
                self.chunks.append(chunk)
                self.chunk_metadata.append({
                    'document_name': document_name,
                    'chunk_index': i,
                    'global_index': start_index + i,
                    'chunk_length': len(chunk)
                })
            
            # Update document mapping
            if document_name not in self.document_mapping:
                self.document_mapping[document_name] = []
            
            self.document_mapping[document_name].extend(
                list(range(start_index, start_index + len(chunks)))
            )
            
            # Rebuild indices
            self._build_indices()
            
        except Exception as e:
            st.error(f"Error adding documents: {str(e)}")
            raise e
    
    def _build_indices(self):
        """Build both FAISS and TF-IDF indices"""
        try:
            if not self.chunks:
                return
            
            with st.spinner("Building search indices..."):
                # Build sentence transformer embeddings and FAISS index
                embeddings = self.embedding_model.encode(
                    self.chunks, 
                    show_progress_bar=True,
                    convert_to_numpy=True
                )
                
                # Create FAISS index
                dimension = embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
                
                # Normalize embeddings for cosine similarity
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                self.faiss_index.add(embeddings.astype('float32'))
                
                # Build TF-IDF matrix
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.chunks)
                
        except Exception as e:
            st.error(f"Error building indices: {str(e)}")
            raise e
    
    def search(self, query, top_k=5, search_type="hybrid", alpha=0.7):
        """
        Search for relevant chunks using different methods.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            search_type: "semantic", "keyword", or "hybrid"
            alpha: Weight for semantic search in hybrid mode (0-1)
        
        Returns:
            Dictionary with chunks, scores, and metadata
        """
        try:
            if not self.chunks:
                return {
                    "chunks": [],
                    "scores": [],
                    "metadata": [],
                    "source_files": []
                }
            
            if search_type == "semantic":
                return self._semantic_search(query, top_k)
            elif search_type == "keyword":
                return self._keyword_search(query, top_k)
            else:  # hybrid
                return self._hybrid_search(query, top_k, alpha)
                
        except Exception as e:
            st.error(f"Error during search: {str(e)}")
            return {
                "chunks": [],
                "scores": [],
                "metadata": [],
                "source_files": []
            }
    
    def _semantic_search(self, query, top_k):
        """Semantic search using sentence transformers and FAISS"""
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search FAISS index
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
        
        # Prepare results
        chunks = [self.chunks[idx] for idx in indices[0]]
        metadata = [self.chunk_metadata[idx] for idx in indices[0]]
        source_files = [meta['document_name'] for meta in metadata]
        
        return {
            "chunks": chunks,
            "scores": scores[0].tolist(),
            "metadata": metadata,
            "source_files": source_files
        }
    
    def _keyword_search(self, query, top_k):
        """Keyword search using TF-IDF"""
        # Transform query
        query_vec = self.tfidf_vectorizer.transform([query])
        
        # Calculate similarity
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]
        
        # Prepare results
        chunks = [self.chunks[idx] for idx in top_indices]
        metadata = [self.chunk_metadata[idx] for idx in top_indices]
        source_files = [meta['document_name'] for meta in metadata]
        
        return {
            "chunks": chunks,
            "scores": top_scores.tolist(),
            "metadata": metadata,
            "source_files": source_files
        }
    
    def _hybrid_search(self, query, top_k, alpha):
        """Hybrid search combining semantic and keyword search"""
        # Get results from both methods
        semantic_results = self._semantic_search(query, top_k * 2)
        keyword_results = self._keyword_search(query, top_k * 2)
        
        # Combine and re-rank results
        combined_scores = {}
        
        # Add semantic scores
        for i, (chunk, score, metadata) in enumerate(zip(
            semantic_results["chunks"],
            semantic_results["scores"],
            semantic_results["metadata"]
        )):
            chunk_id = metadata["global_index"]
            combined_scores[chunk_id] = {
                "chunk": chunk,
                "metadata": metadata,
                "semantic_score": score,
                "keyword_score": 0
            }
        
        # Add keyword scores
        for i, (chunk, score, metadata) in enumerate(zip(
            keyword_results["chunks"],
            keyword_results["scores"],
            keyword_results["metadata"]
        )):
            chunk_id = metadata["global_index"]
            if chunk_id in combined_scores:
                combined_scores[chunk_id]["keyword_score"] = score
            else:
                combined_scores[chunk_id] = {
                    "chunk": chunk,
                    "metadata": metadata,
                    "semantic_score": 0,
                    "keyword_score": score
                }
        
        # Calculate combined scores
        for chunk_id in combined_scores:
            semantic_score = combined_scores[chunk_id]["semantic_score"]
            keyword_score = combined_scores[chunk_id]["keyword_score"]
            combined_scores[chunk_id]["final_score"] = (
                alpha * semantic_score + (1 - alpha) * keyword_score
            )
        
        # Sort by final score and get top k
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x["final_score"],
            reverse=True
        )[:top_k]
        
        # Prepare final results
        chunks = [r["chunk"] for r in sorted_results]
        scores = [r["final_score"] for r in sorted_results]
        metadata = [r["metadata"] for r in sorted_results]
        source_files = [meta["document_name"] for meta in metadata]
        
        return {
            "chunks": chunks,
            "scores": scores,
            "metadata": metadata,
            "source_files": source_files
        }
    
    def get_document_stats(self):
        """Get statistics about stored documents"""
        if not self.chunks:
            return {}
        
        stats = {
            "total_documents": len(self.document_mapping),
            "total_chunks": len(self.chunks),
            "avg_chunk_length": np.mean([len(chunk) for chunk in self.chunks]),
            "document_breakdown": {}
        }
        
        for doc_name, chunk_indices in self.document_mapping.items():
            doc_chunks = [self.chunks[i] for i in chunk_indices]
            stats["document_breakdown"][doc_name] = {
                "chunks": len(doc_chunks),
                "total_length": sum(len(chunk) for chunk in doc_chunks),
                "avg_chunk_length": np.mean([len(chunk) for chunk in doc_chunks])
            }
        
        return stats
    
    def remove_document(self, document_name):
        """Remove a document from the vector store"""
        if document_name not in self.document_mapping:
            return False
        
        # Get chunk indices to remove
        indices_to_remove = set(self.document_mapping[document_name])
        
        # Create new lists without the removed chunks
        new_chunks = []
        new_metadata = []
        
        for i, (chunk, metadata) in enumerate(zip(self.chunks, self.chunk_metadata)):
            if i not in indices_to_remove:
                new_chunks.append(chunk)
                new_metadata.append({
                    **metadata,
                    "global_index": len(new_chunks) - 1  # Update global index
                })
        
        # Update instance variables
        self.chunks = new_chunks
        self.chunk_metadata = new_metadata
        
        # Remove from document mapping
        del self.document_mapping[document_name]
        
        # Update other document mappings
        for doc_name in self.document_mapping:
            updated_indices = []
            for old_idx in self.document_mapping[doc_name]:
                if old_idx not in indices_to_remove:
                    # Calculate new index
                    new_idx = old_idx - sum(1 for removed_idx in indices_to_remove if removed_idx < old_idx)
                    updated_indices.append(new_idx)
            self.document_mapping[doc_name] = updated_indices
        
        # Rebuild indices if there are still documents
        if self.chunks:
            self._build_indices()
        else:
            self.faiss_index = None
            self.tfidf_matrix = None
        
        return True
    
    def save_index(self, filepath):
        """Save the vector store to disk"""
        try:
            data = {
                "chunks": self.chunks,
                "chunk_metadata": self.chunk_metadata,
                "document_mapping": self.document_mapping,
                "embedding_model_name": self.embedding_model_name
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            # Save FAISS index separately
            if self.faiss_index is not None:
                faiss.write_index(self.faiss_index, filepath + ".faiss")
            
            return True
        except Exception as e:
            st.error(f"Error saving index: {str(e)}")
            return False
    
    def load_index(self, filepath):
        """Load the vector store from disk"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.chunks = data["chunks"]
            self.chunk_metadata = data["chunk_metadata"]
            self.document_mapping = data["document_mapping"]
            self.embedding_model_name = data["embedding_model_name"]
            
            # Load models
            self._load_models()
            
            # Load FAISS index
            faiss_path = filepath + ".faiss"
            try:
                self.faiss_index = faiss.read_index(faiss_path)
            except:
                st.warning("FAISS index not found, rebuilding...")
                self._build_indices()
            
            # Rebuild TF-IDF matrix
            if self.chunks:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.chunks)
            
            return True
        except Exception as e:
            st.error(f"Error loading index: {str(e)}")
            return False