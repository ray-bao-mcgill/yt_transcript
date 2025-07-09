#!/usr/bin/env python3
"""
Test script for BERT Word Embedding Clustering Analyzer
Tests word clustering functionality within single transcripts
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bert_word_clustering_analyzer import BERTWordClusteringAnalyzer

def convert_ndarrays_to_lists(obj):
    """Recursively convert numpy arrays and numpy scalar types to lists/native types for JSON serialization"""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.generic,)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_ndarrays_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarrays_to_lists(v) for v in obj]
    else:
        return obj

def test_word_clustering():
    """Test the word clustering analyzer with sample text"""
    
    print("Testing BERT Word Embedding Clustering Analyzer")
    print("=" * 50)
    
    # Initialize analyzer
    try:
        analyzer = BERTWordClusteringAnalyzer()
        print("✓ BERT Word Clustering Analyzer initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize analyzer: {e}")
        return
    
    # Sample transcript for testing
    sample_transcript = """
    The government should reduce regulations and allow free market principles to drive economic growth. 
    Lower taxes will stimulate business investment and create jobs for hardworking Americans. 
    We need to strengthen our borders and enforce immigration laws. 
    The rule of law is fundamental to our democracy and national security. 
    Traditional family values are the foundation of a strong society. 
    We must protect religious freedom and support pro-life policies.
    The economy is showing strong growth with unemployment at historic lows.
    Small businesses are the backbone of our economy and deserve support.
    We need to invest in infrastructure and create jobs for working families.
    The American dream is alive and well for those willing to work hard.
    """
    
    print(f"\nSample transcript length: {len(sample_transcript)} characters")
    
    # Test word embedding extraction
    print("\n1. Testing word embedding extraction...")
    try:
        embedding_results = analyzer.extract_word_embeddings(sample_transcript, min_word_length=3)
        print(f"✓ Extracted embeddings for {embedding_results['embedded_words']} words")
        print(f"  Total significant words found: {embedding_results['total_words']}")
        
        # Show some example words
        word_embeddings = embedding_results['word_embeddings']
        example_words = list(word_embeddings.keys())[:10]
        print(f"  Example words: {', '.join(example_words)}")
        
    except Exception as e:
        print(f"✗ Word embedding extraction failed: {e}")
        return
    
    # Test clustering
    print("\n2. Testing word clustering...")
    try:
        cluster_results = analyzer.cluster_word_embeddings(
            word_embeddings, 
            method='kmeans', 
            n_clusters=5
        )
        print(f"✓ Clustering completed successfully")
        print(f"  Number of clusters: {cluster_results['n_clusters']}")
        print(f"  Clustering method: {cluster_results['method']}")
        if cluster_results.get('silhouette_score'):
            print(f"  Silhouette score: {cluster_results['silhouette_score']:.3f}")
        
    except Exception as e:
        print(f"✗ Clustering failed: {e}")
        return
    
    # Test cluster analysis
    print("\n3. Testing cluster analysis...")
    try:
        cluster_analysis = analyzer.analyze_word_clusters(
            cluster_results, 
            embedding_results['word_contexts']
        )
        print(f"✓ Cluster analysis completed successfully")
        print(f"  Number of analyzed clusters: {len(cluster_analysis)}")
        
        # Show details for each cluster
        for cluster_name, cluster_data in cluster_analysis.items():
            print(f"\n  {cluster_name}:")
            print(f"    Size: {cluster_data['size']} words")
            print(f"    Avg word length: {cluster_data['avg_length']:.1f}")
            print(f"    Political terms: {len(cluster_data['political_terms'])}")
            print(f"    Sample words: {', '.join(cluster_data['words'][:5])}")
        
    except Exception as e:
        print(f"✗ Cluster analysis failed: {e}")
        return
    
    # Test visualizations
    print("\n4. Testing visualization creation...")
    try:
        visualizations = analyzer.create_word_visualizations(cluster_results, cluster_analysis)
        
        if 'error' in visualizations:
            print(f"✗ Visualization creation failed: {visualizations['error']}")
        else:
            print("✓ Visualizations created successfully")
            available_viz = [k for k, v in visualizations.items() if v and 'error' not in str(v)]
            print(f"  Available visualizations: {', '.join(available_viz)}")
        
    except Exception as e:
        print(f"✗ Visualization creation failed: {e}")
    
    # Test complete analysis pipeline
    print("\n5. Testing complete analysis pipeline...")
    try:
        results = analyzer.analyze_single_transcript(
            sample_transcript, 
            method='kmeans', 
            n_clusters=5
        )
        
        if 'error' in results:
            print(f"✗ Complete analysis failed: {results['error']}")
        else:
            print("✓ Complete analysis pipeline successful")
            print(f"  Transcript length: {results['transcript_length']}")
            print(f"  Embedded words: {results['embedding_results']['embedded_words']}")
            print(f"  Clusters found: {results['cluster_results']['n_clusters']}")
            print(f"  Silhouette score: {results.get('silhouette_score', 'N/A')}")
        
    except Exception as e:
        print(f"✗ Complete analysis failed: {e}")
    
    print("\n" + "=" * 50)
    print("Word clustering test completed!")

def test_different_clustering_methods():
    """Test different clustering methods"""
    
    print("\nTesting Different Clustering Methods")
    print("=" * 40)
    
    try:
        analyzer = BERTWordClusteringAnalyzer()
        
        # Sample text
        sample_text = """
        The government should reduce regulations and allow free market principles to drive economic growth. 
        Lower taxes will stimulate business investment and create jobs for hardworking Americans. 
        We need to strengthen our borders and enforce immigration laws. 
        The rule of law is fundamental to our democracy and national security.
        """
        
        # Test different methods
        methods = ['kmeans', 'dbscan', 'agglomerative']
        
        for method in methods:
            print(f"\nTesting {method.upper()} clustering...")
            try:
                results = analyzer.analyze_single_transcript(
                    sample_text, 
                    method=method, 
                    n_clusters=4
                )
                
                if 'error' not in results:
                    print(f"  ✓ {method.upper()} successful")
                    print(f"    Clusters: {results['cluster_results']['n_clusters']}")
                    print(f"    Silhouette: {results.get('silhouette_score', 'N/A')}")
                else:
                    print(f"  ✗ {method.upper()} failed: {results['error']}")
                    
            except Exception as e:
                print(f"  ✗ {method.upper()} failed: {e}")
        
    except Exception as e:
        print(f"✗ Method testing failed: {e}")

if __name__ == "__main__":
    print("BERT Word Embedding Clustering Analyzer Test Suite")
    print("=" * 60)
    
    # Run basic tests
    test_word_clustering()
    
    # Run method comparison tests
    test_different_clustering_methods()
    
    print("\nAll tests completed!") 